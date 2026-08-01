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

class Section1Scene(TeachingScene):
    def construct(self):
        title_str = "The Problem: The Noisy Cosmic Message"
        lecture_strs = [
            "Meet Byte, sending data from Mars to Earth.",
            "Space radiation can flip bits during transmission.",
            "Errors turn clear messages into garbled noise."
        ]
        self.setup_layout(title_str, lecture_strs)

        # Assets - Using provided SVG paths from storyboard/issues
        mars = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mars.svg").set_color(RED)
        earth = Circle(color=BLUE, fill_opacity=1) # No SVG provided for Earth in storyboard
        byte_robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        radiation = Triangle(color=YELLOW, fill_opacity=1) # No SVG provided for radiation

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Place planets in designated areas
        self.place_in_area(mars, "A1", "B2", scale_factor=0.8)
        self.place_in_area(earth, "E5", "F6", scale_factor=0.8)
        
        # Issue 27: Position byte_robot at A3 to avoid overlap with Mars area
        self.place_at_grid(byte_robot, "A3", scale_factor=0.4)
        
        # Issue 27: Position bits at B3
        bits = VGroup(*[Text(b, font_size=36, color=YELLOW) for b in "1011"])
        bits.arrange(RIGHT, buff=0.1)
        self.place_at_grid(bits, "B3", scale_factor=0.8)
        
        self.play(FadeIn(mars), FadeIn(earth), FadeIn(byte_robot))
        self.play(Write(bits))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(RED)
        
        # Issue 28: Position radiation at C4 to align with transmission path
        self.place_at_grid(radiation, "C4", scale_factor=0.5)
        
        self.play(FadeIn(radiation))
        
        # Travel towards Earth, hit by radiation at C4
        self.play(bits.animate.move_to(self.grid["C4"]), run_time=1.5)
        
        # Bit flip animation at C4
        flash = Flash(self.grid["C4"], color=RED, num_lines=8)
        
        # Third bit is index 2 ('1')
        old_bit = bits[2]
        new_bit = Text("0", font_size=36, color=RED)
        # Position new bit exactly where the old bit is currently
        new_bit.move_to(old_bit.get_center())
        
        self.play(flash)
        self.play(
            FadeOut(old_bit, shift=UP),
            FadeIn(new_bit, shift=UP)
        )
        # Update VGroup structure for subsequent movement
        bits.remove(old_bit)
        bits.insert(2, new_bit)
        
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(ORANGE)
        
        # Finish journey to Earth area center
        earth_center = (self.grid["E5"] + self.grid["F6"]) / 2
        self.play(bits.animate.move_to(earth_center), run_time=1.5)
        
        # Issue 29: Position q_mark at D6 with appropriate scale
        q_mark = Text("?", font_size=48, color=ORANGE)
        self.place_at_grid(q_mark, "D6", scale_factor=0.8)
        
        self.play(Write(q_mark))
        self.play(Indicate(earth, color=ORANGE))
        
        self.wait(2)
