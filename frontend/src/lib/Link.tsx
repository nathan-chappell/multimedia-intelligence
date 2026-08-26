import { type AnchorHTMLAttributes, type MouseEvent } from "react";

import { navigate } from "./navigation";

type LinkProps = Omit<AnchorHTMLAttributes<HTMLAnchorElement>, "href"> & {
  href: string;
};

export function Link({ href, onClick, target, ...props }: LinkProps) {
  function handleClick(event: MouseEvent<HTMLAnchorElement>) {
    onClick?.(event);
    if (
      event.defaultPrevented
      || event.button !== 0
      || event.metaKey
      || event.ctrlKey
      || event.shiftKey
      || event.altKey
      || target === "_blank"
    ) return;

    const destination = new URL(href, window.location.href);
    if (destination.origin !== window.location.origin) return;
    event.preventDefault();
    navigate(destination.href);
  }

  return <a {...props} href={href} target={target} onClick={handleClick} />;
}
