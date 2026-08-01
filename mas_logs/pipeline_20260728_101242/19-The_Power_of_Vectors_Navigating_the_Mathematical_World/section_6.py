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
        self.setup_layout("Summary and Real-world Impact", [
            "Vectors elegantly combine magnitude with precise direction.",
            "They drive physics engines and modern computer graphics.",
            "Master these tools to navigate the mathematical world."
        ])
        
        # Colors (Nord Palette inspired)
        COLOR_MAG = "#88C0D0"
        COLOR_DIR = "#BF616A"
        COLOR_SCL = "#A3BE8C"
        COLOR_ADD = "#EBCB8B"
        COLOR_GRAVITY = "#D08770"
        COLOR_JUMP = "#B48EAD"
        COLOR_RESULTANT = "#EBCB8B"

        # === Animation for Lecture Line 1 ===
        # Line 1: Vectors elegantly combine magnitude with precise direction.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Create icons for fundamental vector concepts
        mag_icon = VGroup(
            Line(LEFT*0.3, RIGHT*0.3, color=COLOR_MAG),
            Line(UP*0.1, DOWN*0.1, color=COLOR_MAG).shift(LEFT*0.3),
            Line(UP*0.1, DOWN*0.1, color=COLOR_MAG).shift(RIGHT*0.3)
        )
        mag_label = Text("Magnitude", font_size=16, color=COLOR_MAG).next_to(mag_icon, DOWN, buff=0.1)
        mag_group = VGroup(mag_icon, mag_label)
        
        dir_icon = Arrow(LEFT*0.3, RIGHT*0.3, color=COLOR_DIR, buff=0)
        dir_label = Text("Direction", font_size=16, color=COLOR_DIR).next_to(dir_icon, DOWN, buff=0.1)
        dir_group = VGroup(dir_icon, dir_label)
        
        scl_icon = VGroup(
            Arrow(LEFT*0.3, RIGHT*0.3, color=COLOR_SCL, buff=0),
            DashedLine(LEFT*0.3, RIGHT*0.6, color=COLOR_SCL).shift(DOWN*0.1)
        )
        scl_label = Text("Scaling", font_size=16, color=COLOR_SCL).next_to(scl_icon, DOWN, buff=0.1)
        scl_group = VGroup(scl_icon, scl_label)
        
        add_icon = VGroup(
            Arrow(ORIGIN, RIGHT*0.3, color=COLOR_ADD, buff=0),
            Arrow(RIGHT*0.3, RIGHT*0.3+UP*0.3, color=COLOR_ADD, buff=0),
            Line(ORIGIN, RIGHT*0.3+UP*0.3, color=WHITE, stroke_width=2)
        )
        add_label = Text("Addition", font_size=16, color=COLOR_ADD).next_to(add_icon, DOWN, buff=0.1)
        add_group = VGroup(add_icon, add_label)

        # Place icons in the grid per VideoCritic feedback
        # Issue 35: mag_group at 'B4', dir_group at 'B5'
        self.place_at_grid(mag_group, "B4")
        self.place_at_grid(dir_group, "B5")
        # Issue 36: scl_group at 'C4', add_group at 'C5'
        self.place_at_grid(scl_group, "C4")
        self.place_at_grid(add_group, "C5")

        self.play(FadeIn(mag_group), FadeIn(dir_group), FadeIn(scl_group), FadeIn(add_group))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Line 2: They drive physics engines and modern computer graphics.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW),
            FadeOut(mag_group), FadeOut(dir_group), FadeOut(scl_group), FadeOut(add_group)
        )
        
        # Issue 22: Character (Asset-based)
        character = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/character.png")
        # Issue 37: Move character to 'E4'
        self.place_at_grid(character, "E4", scale_factor=0.6)
        
        # Define Gravity and Jump vectors
        # Gravity pulls down to F4 (from E4)
        gravity_vec = Arrow(self.grid["E4"], self.grid["F4"], color=COLOR_GRAVITY, buff=0)
        gravity_label = Text("Gravity", font_size=18, color=COLOR_GRAVITY).next_to(gravity_vec, DOWN, buff=0.1)
        
        # Jump pushes up/right to B5 (from E4: up 3 rows, right 1 col)
        jump_vec = Arrow(self.grid["E4"], self.grid["B5"], color=COLOR_JUMP, buff=0)
        jump_label = Text("Jump", font_size=18, color=COLOR_JUMP).next_to(jump_vec, UP, buff=0.1)
        
        self.play(FadeIn(character))
        self.play(GrowArrow(gravity_vec), Write(gravity_label))
        self.play(GrowArrow(jump_vec), Write(jump_label))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Line 3: Master these tools to navigate the mathematical world.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Resultant Vector path (Jump + Gravity)
        # Vector Sum from E4: (F4-E4) + (B5-E4) leads to position C5 (up 2, right 1)
        target_pos = self.grid["C5"]
        start_pos = self.grid["E4"]
        
        resultant_vec = Arrow(start_pos, target_pos, color=COLOR_RESULTANT, buff=0)
        resultant_label = Text("Resultant Path", font_size=18, color=COLOR_RESULTANT).next_to(resultant_vec, RIGHT, buff=0.2)
        
        self.play(
            FadeOut(gravity_label), 
            FadeOut(jump_label),
            FadeOut(gravity_vec),
            FadeOut(jump_vec),
            GrowArrow(resultant_vec),
            Write(resultant_label)
        )
        
        # Animate character moving along the resultant vector path
        # The vector and label move with the character to visualize displacement
        self.play(
            character.animate.move_to(target_pos),
            resultant_vec.animate.shift(target_pos - start_pos),
            resultant_label.animate.shift(target_pos - start_pos),
            run_time=2,
            rate_func=smooth
        )
        
        self.wait(3)
