from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section7Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        lecture_lines = [
            "We begin with a canvas of pure random noise.",
            "CLIP translates our prompt into a guiding vector.",
            "The U-Net iteratively refines the noise into art.",
            "Order emerges from chaos through mathematics and language."
        ]
        self.setup_layout("The Final Pipeline: Inference", lecture_lines)

        # Colors
        prompt_color = "#FFFFFF"
        clip_color = "#FFFF00"
        unet_color = "#BB88FF"
        flow_color = "#00FF00"
        highlight_color = "#FFFF00"
        
        # Asset path
        watch_asset_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/watch.svg"

        # Noise simulation
        def create_noise(opacity=1.0):
            noise_group = VGroup()
            for _ in range(15):
                for _ in range(15):
                    square = Square(side_length=0.1, stroke_width=0, fill_opacity=opacity)
                    square.set_fill(color=interpolate_color(WHITE, BLACK, np.random.rand()))
                    noise_group.add(square)
            noise_group.arrange_in_grid(rows=15, cols=15, buff=0)
            return noise_group

        # Labels/Mobjects
        xt_label = Text("xT (Noise)", font_size=20)
        x0_label = Text("x0 (Image)", font_size=20)
        prompt_text = Text('"Vintage Steampunk Watch"', font_size=24, color=prompt_color)
        
        clip_box = VGroup(
            RoundedRectangle(corner_radius=0.2, height=0.8, width=1.6, color=clip_color),
            Text("CLIP", font_size=22, color=clip_color)
        )
        unet_box = VGroup(
            RoundedRectangle(corner_radius=0.2, height=0.8, width=1.6, color=unet_color),
            Text("U-Net", font_size=22, color=unet_color)
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(highlight_color)
        noise_canvas = create_noise()
        self.place_in_area(noise_canvas, "C2", "E5", scale_factor=2.0)
        
        # [Issue 52/60] Positioning labels and prompt
        self.place_in_area(xt_label, "B3", "B4", scale_factor=1.0)
        self.place_in_area(prompt_text, "A3", "A4", scale_factor=1.0)
        
        self.play(FadeIn(noise_canvas), FadeIn(xt_label), FadeIn(prompt_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(highlight_color)
        
        # [Issue 52/60] clip_box at B2, scale 0.8
        self.place_at_grid(clip_box, "B2", scale_factor=0.8)
        
        # Vector from CLIP to the guiding region
        clip_vector = Arrow(clip_box.get_right(), self.grid["B3"], color=clip_color, buff=0.1)
        vector_label = Text("Guiding Vector", font_size=16, color=clip_color).next_to(clip_vector, UP, buff=0.1)

        self.play(FadeIn(clip_box))
        self.play(GrowArrow(clip_vector), Write(vector_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(highlight_color)
        
        self.place_at_grid(unet_box, "B5", scale_factor=0.8)
        unet_arrow = Arrow(self.grid["B4"], unet_box.get_left(), color=unet_color, buff=0.1)
        feedback_loop = Arc(radius=0.3, start_angle=0, angle=TAU*0.75, color=unet_color).add_tip()
        feedback_loop.next_to(unet_box, RIGHT, buff=0.1)
        
        self.play(FadeIn(unet_box), GrowArrow(unet_arrow))
        
        # Refinement Iteration Simulation
        for label_text, opacity in zip(["x50", "x40", "x10"], [0.8, 0.5, 0.2]):
            step_noise = create_noise(opacity=opacity)
            self.place_in_area(step_noise, "C2", "E5", scale_factor=2.0)
            iter_label = Text(label_text, font_size=20, color=WHITE).next_to(xt_label, RIGHT, buff=0.5)
            self.play(
                noise_canvas.animate.become(step_noise),
                FadeIn(iter_label, run_time=0.2),
                Rotate(feedback_loop, angle=TAU, about_point=feedback_loop.get_center()),
                run_time=0.8
            )
            self.play(FadeOut(iter_label, run_time=0.2))

        # [Issue 34/60] Final Image Emerges using Asset
        watch_image = SVGMobject(watch_asset_path)
        self.place_in_area(watch_image, "C2", "E5", scale_factor=1.2)
        # [Issue 53/60] x0_label in area F3-F4
        self.place_in_area(x0_label, "F3", "F4", scale_factor=1.0)
        
        self.play(
            ReplacementTransform(noise_canvas, watch_image),
            ReplacementTransform(xt_label, x0_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(highlight_color)
        
        # Clear specific parts for final pipeline summary
        self.play(
            FadeOut(clip_vector), FadeOut(vector_label), FadeOut(unet_arrow), 
            FadeOut(feedback_loop), FadeOut(watch_image), FadeOut(x0_label),
            FadeOut(prompt_text), FadeOut(clip_box), FadeOut(unet_box)
        )

        # [Issue 51/60] Flow Chart: [Text] -> [CLIP] -> [U-Net] -> [Image]
        f_text = Text("Text", font_size=22, color=flow_color)
        f_clip = Text("CLIP", font_size=22, color=flow_color)
        f_unet = Text("U-Net", font_size=22, color=flow_color)
        
        # Asset integration for flow chart Image
        f_img_svg = SVGMobject(watch_asset_path).set_color(flow_color)
        f_img_label = Text("Image", font_size=20, color=flow_color)
        f_img_group = VGroup(f_img_svg, f_img_label).arrange(DOWN, buff=0.1)
        
        # Apply positioning from Issue 51
        self.place_at_grid(f_text, "B2", scale_factor=1.0)
        self.place_at_grid(f_clip, "B5", scale_factor=1.0)
        self.place_at_grid(f_unet, "D2", scale_factor=1.0)
        self.place_at_grid(f_img_group, "D5", scale_factor=0.6)
        
        # Connectors for zigzag layout
        a1 = Arrow(f_text.get_right(), f_clip.get_left(), color=flow_color, buff=0.2)
        a2 = Arrow(f_clip.get_bottom(), f_unet.get_top(), color=flow_color, buff=0.2)
        a3 = Arrow(f_unet.get_right(), f_img_group.get_left(), color=flow_color, buff=0.2)
        
        pipeline = VGroup(f_text, a1, f_clip, a2, f_unet, a3, f_img_group)
        
        self.play(Write(pipeline))
        self.wait(2)
