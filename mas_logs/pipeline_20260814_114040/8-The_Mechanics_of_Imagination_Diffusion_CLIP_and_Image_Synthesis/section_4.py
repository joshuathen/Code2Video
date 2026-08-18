from manim import *
import numpy as np
import os

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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Synthesis combines CLIP and diffusion models.",
            "CLIP provides a textual target direction.",
            "The diffusion model iteratively refines the image.",
            "Loss is minimized between noise and text.",
            "The loop creates the final generated art."
        ]
        self.setup_layout("Synthesis: The Feedback Loop", lecture_lines)
        
        # Assets: Check file existence before loading to avoid ParseError
        comp_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg"
        cam_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg"
        
        if os.path.exists(comp_path):
            computer_icon = SVGMobject(comp_path)
        else:
            computer_icon = Dot(color=BLUE)
            
        if os.path.exists(cam_path):
            camera_icon = SVGMobject(cam_path)
        else:
            camera_icon = Dot(color=RED)
        
        clip_box = Rectangle(width=2, height=1, color=BLUE).set_fill(BLUE, opacity=0.3)
        diff_box = Rectangle(width=2, height=1, color=PURPLE).set_fill(PURPLE, opacity=0.3)
        self.place_at_grid(clip_box, "B3", scale_factor=0.7)
        self.place_at_grid(diff_box, "D3", scale_factor=0.7)
        
        clip_text = Text("CLIP", font_size=20).move_to(clip_box)
        diff_text = Text("Diffusion", font_size=20).move_to(diff_box)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF8C00")
        self.play(Create(clip_box), Write(clip_text), Create(diff_box), Write(diff_text), FadeIn(self.place_at_grid(computer_icon, "B5", scale_factor=0.5)))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF8C00")
        prompt = Text("Prompt", font_size=18, color=YELLOW)
        self.place_at_grid(prompt, "B2")
        arrow = Arrow(prompt.get_right(), clip_box.get_left(), color=WHITE)
        self.play(FadeIn(prompt), GrowArrow(arrow))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF8C00")
        image_node = Circle(radius=0.4, color=WHITE).set_fill(WHITE, opacity=0.2)
        self.place_at_grid(image_node, "D2")
        self.play(FadeIn(image_node), FadeIn(self.place_at_grid(camera_icon, "D6", scale_factor=0.5)))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#32CD32")
        loss_text = Text("Loss = min(d)", font_size=16, color="#32CD32")
        self.place_at_grid(loss_text, "E3", scale_factor=0.6)
        self.play(Write(loss_text))
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF8C00")
        target_icon = Star(color=YELLOW).set_fill(YELLOW, opacity=0.8)
        self.place_at_grid(target_icon, "D5", scale_factor=0.5)
        self.play(FadeIn(target_icon))
        self.wait(1)
