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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup layout
        lines = [
            "Repeating these steps makes the target state dominate.",
            "The correct answer reaches nearly 100% probability.",
            "We then measure the system to find our item."
        ]
        self.setup_layout("Measurement and Convergence", lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        # Symbols for Oracle and Diffusion
        oracle_box = Rectangle(width=1.5, height=0.8, color=WHITE)
        oracle_text = Text("Oracle", font_size=20, color=WHITE)
        oracle_group = VGroup(oracle_box, oracle_text)
        
        diffusion_box = Rectangle(width=1.5, height=0.8, color=WHITE)
        diffusion_text = Text("Diffusion", font_size=20, color=WHITE)
        diffusion_group = VGroup(diffusion_box, diffusion_text)
        
        # Fix: Issue 49 & 57 (Move left to avoid clipping)
        self.place_at_grid(oracle_group, 'B2', scale_factor=0.8)
        self.place_at_grid(diffusion_group, 'D2', scale_factor=0.8)
        
        # Repeat loop symbol
        loop_arrow = CurvedArrow(
            self.grid["D2"] + RIGHT*0.7, 
            self.grid["B2"] + RIGHT*0.7, 
            angle=-PI, 
            color=WHITE
        )
        # Fix: Issue 50 & 57 (Move label to avoid clipping)
        loop_label = Text("Repeat", font_size=18, color=WHITE)
        self.place_at_grid(loop_label, 'C3', scale_factor=0.7)
        
        self.play(Create(oracle_group), Create(diffusion_group))
        self.play(Create(loop_arrow), Write(loop_label))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(GRAY), self.lecture[1].animate.set_color("#FFD700"))
        self.play(FadeOut(oracle_group), FadeOut(diffusion_group), FadeOut(loop_arrow), FadeOut(loop_label))
        
        # Bar Chart representation
        # 4 bars, index 2 is target
        bar_width = 0.6
        bars = VGroup(*[
            Rectangle(width=bar_width, height=0.5, fill_opacity=0.6, color=BLUE_E)
            for _ in range(4)
        ]).arrange(RIGHT, buff=0.3)
        
        # Highlight target bar in GOLD
        bars[2].set_color("#FFD700")
        bars[2].set_fill("#FFD700", opacity=0.8)
        
        self.place_in_area(bars, "B2", "E5")
        
        self.play(Create(bars))
        self.wait(1)
        
        # Convergence animation: target grows, others vanish
        self.play(
            bars[2].animate.stretch_to_fit_height(4).move_to(bars[2].get_bottom() + UP*2),
            bars[0].animate.stretch_to_fit_height(0.05).move_to(bars[0].get_bottom() + UP*0.025),
            bars[1].animate.stretch_to_fit_height(0.05).move_to(bars[1].get_bottom() + UP*0.025),
            bars[3].animate.stretch_to_fit_height(0.05).move_to(bars[3].get_bottom() + UP*0.025),
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(GRAY), self.lecture[2].animate.set_color(WHITE))
        
        # Reveal Pixel star and Search Complete text
        # Fix: Issue 33 & 57 (Use Asset)
        pixel_star = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/star.svg").set_color("#FFD700")
        complete_text = Text("Search Complete", font_size=32, color="#FFFFFF")
        
        self.place_at_grid(pixel_star, "C3", scale_factor=0.8)
        # Fix: Issue 51 & 57 (Prevent text clipping)
        self.place_in_area(complete_text, 'E1', 'E3', scale_factor=0.8)
        
        self.play(FadeOut(bars))
        self.play(DrawBorderThenFill(pixel_star))
        self.play(Write(complete_text))
        
        self.wait(3)
