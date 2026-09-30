import type { ReactNode, SVGProps } from 'react';

type IconProps = SVGProps<SVGSVGElement>;

function Icon({ children, ...props }: IconProps & { children: ReactNode }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      {children}
    </svg>
  );
}

export const IconPlus = (p: IconProps) => (
  <Icon {...p}>
    <path d="M12 5v14M5 12h14" />
  </Icon>
);

export const IconMenu = (p: IconProps) => (
  <Icon {...p}>
    <path d="M4 6h16M4 12h16M4 18h16" />
  </Icon>
);

export const IconArrowUp = (p: IconProps) => (
  <Icon {...p} strokeWidth={2.2}>
    <path d="M12 19V5M6 11l6-6 6 6" />
  </Icon>
);

export const IconStop = (p: IconProps) => (
  <Icon {...p} stroke="none" fill="currentColor">
    <rect x="6" y="6" width="12" height="12" rx="2.5" />
  </Icon>
);

export const IconX = (p: IconProps) => (
  <Icon {...p}>
    <path d="M6 6l12 12M18 6L6 18" />
  </Icon>
);

/** 摊开的手册：品牌标识与助手头像 */
export const IconManual = (p: IconProps) => (
  <Icon {...p}>
    <path d="M12 6.5C10 5 7.5 4.5 4 5v13c3.5-.5 6 0 8 1.5 2-1.5 4.5-2 8-1.5V5c-3.5-.5-6 0-8 1.5z" />
    <path d="M12 6.5v13" />
  </Icon>
);

export const IconFileText = (p: IconProps) => (
  <Icon {...p}>
    <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
    <path d="M14 3v5h5M9 13h6M9 17h4" />
  </Icon>
);

export const IconTable = (p: IconProps) => (
  <Icon {...p}>
    <rect x="3.5" y="4.5" width="17" height="15" rx="2" />
    <path d="M3.5 10h17M3.5 15h17M9.5 10v9.5" />
  </Icon>
);

export const IconLayers = (p: IconProps) => (
  <Icon {...p}>
    <path d="M12 3.5l8.5 4.5-8.5 4.5L3.5 8z" />
    <path d="M3.5 12.5l8.5 4.5 8.5-4.5" />
  </Icon>
);

export const IconSparkle = (p: IconProps) => (
  <Icon {...p}>
    <path d="M12 3.5l1.9 5.1 5.1 1.9-5.1 1.9-1.9 5.1-1.9-5.1L5 10.5l5.1-1.9z" />
    <path d="M18.5 16.5l.7 1.8 1.8.7-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7z" />
  </Icon>
);

export const IconCheck = (p: IconProps) => (
  <Icon {...p} strokeWidth={2.2}>
    <path d="M5 12.5l4.5 4.5L19 7.5" />
  </Icon>
);

export const IconShieldCheck = (p: IconProps) => (
  <Icon {...p}>
    <path d="M12 3l7.5 3v5.5c0 4.4-3.1 8.2-7.5 9.5-4.4-1.3-7.5-5.1-7.5-9.5V6z" />
    <path d="M9 12l2.2 2.2L15.5 10" />
  </Icon>
);

export const IconShieldAlert = (p: IconProps) => (
  <Icon {...p}>
    <path d="M12 3l7.5 3v5.5c0 4.4-3.1 8.2-7.5 9.5-4.4-1.3-7.5-5.1-7.5-9.5V6z" />
    <path d="M12 8.5v4.5M12 16.2v.1" />
  </Icon>
);

export const IconAlert = (p: IconProps) => (
  <Icon {...p}>
    <path d="M10.3 4.3L2.8 17.5A2 2 0 0 0 4.5 20.5h15a2 2 0 0 0 1.7-3L13.7 4.3a2 2 0 0 0-3.4 0z" />
    <path d="M12 9.5v4M12 17v.1" />
  </Icon>
);

export const IconCopy = (p: IconProps) => (
  <Icon {...p}>
    <rect x="8.5" y="8.5" width="12" height="12" rx="2" />
    <path d="M15.5 8.5V5.5a2 2 0 0 0-2-2h-8a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h3" />
  </Icon>
);

export const IconRefresh = (p: IconProps) => (
  <Icon {...p}>
    <path d="M20 11.5A8 8 0 1 0 17.7 17" />
    <path d="M20 5v6.5h-6.5" />
  </Icon>
);

export const IconChevronDown = (p: IconProps) => (
  <Icon {...p}>
    <path d="M6 9l6 6 6-6" />
  </Icon>
);

export const IconArrowRight = (p: IconProps) => (
  <Icon {...p}>
    <path d="M5 12h14M13 6l6 6-6 6" />
  </Icon>
);

export const IconChat = (p: IconProps) => (
  <Icon {...p}>
    <path d="M5 5h14a1.5 1.5 0 0 1 1.5 1.5v9A1.5 1.5 0 0 1 19 17h-8.5L6 20.5V17H5a1.5 1.5 0 0 1-1.5-1.5v-9A1.5 1.5 0 0 1 5 5z" />
  </Icon>
);

export function LogoMark({ size = 'md' }: { size?: 'sm' | 'md' | 'lg' }) {
  const box = { sm: 'h-7 w-7 rounded-lg', md: 'h-9 w-9 rounded-xl', lg: 'h-14 w-14 rounded-2xl' }[size];
  const icon = { sm: 'h-4 w-4', md: 'h-5 w-5', lg: 'h-7 w-7' }[size];
  return (
    <div
      className={`${box} flex shrink-0 items-center justify-center bg-gradient-to-br from-teal-500 to-teal-700 text-white shadow-sm shadow-teal-700/20`}
    >
      <IconManual className={icon} />
    </div>
  );
}
