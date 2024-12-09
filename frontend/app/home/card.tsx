import React from "react";
import { PinContainer } from "../components/ui/3d-pin";

interface AnimatedPinDemoProps {
  title: string;
  linkhref: string;
  modelname: string;
  description: string;
  bgimg:string;
}

export function AnimatedPinDemo({ title, linkhref, modelname, description, bgimg }: AnimatedPinDemoProps) {
  return (
    <div className="h-[40rem] w-full flex items-center justify-center">
      <PinContainer title={title} href={linkhref}>
        <div className="flex basis-full flex-col p-4 tracking-tight text-slate-100/50 sm:basis-1/2 w-[20rem] h-[20rem]">
          <h3 className="max-w-xs !pb-2 !m-0 font-bold text-base text-slate-100">
            {modelname}
          </h3>
          <div className="text-base !m-0 !p-0 font-normal">
            <span className="text-slate-500">
              {description}
            </span>
          </div>
          <div className="flex flex-1 w-full rounded-lg mt-4" style={{ backgroundImage: `url(${bgimg})` }}></div>
        </div>
      </PinContainer>
    </div>
  );
}
