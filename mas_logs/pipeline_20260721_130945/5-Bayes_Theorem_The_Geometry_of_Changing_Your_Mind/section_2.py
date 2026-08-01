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

class Section2Scene(TeachingScene):
    def construct(self):
        # Section Title and Lecture Lines
        title_text = "The Scenario: Prior Probabilities"
        lecture_lines = [
            "Meet a robot detective hunting for a Glitch-Bot.",
            "Only twenty percent of bots in this factory glitch.",
            "We split our square to show this prior belief."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Bounding box for the factory (using rows C to E and columns 2 to 5)
        # Dimensions: 3.0 wide, 2.0 high.
        factory_outline = Rectangle(width=3.0, height=2.0, color=WHITE, stroke_width=2)
        self.place_in_area(factory_outline, 'C2', 'E5')
        
        # Descriptive label for the whole factory
        factory_label = Text("Robot Factory (Total Bots)", font_size=22, color=WHITE)
        # RESOLVING ISSUE 27: Center factory_label relative to the outline (B2-B5)
        self.place_in_area(factory_label, 'B2', 'B5', scale_factor=0.8)
        
        self.play(Create(factory_outline), Write(factory_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight Line 2 in red to match the "Glitch-Bot" color
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#E74C3C")
        )
        
        # Divide the 3.0 unit width into 20% (0.6) and 80% (2.4)
        # Colors: red (#E74C3C) and green (#2ECC71)
        red_rect = Rectangle(
            width=0.6, height=2.0, 
            fill_color="#E74C3C", fill_opacity=0.8, 
            stroke_width=1, stroke_color=WHITE
        )
        green_rect = Rectangle(
            width=2.4, height=2.0, 
            fill_color="#2ECC71", fill_opacity=0.8, 
            stroke_width=1, stroke_color=WHITE
        )
        
        # VGroup for simultaneous placement using the 6x6 grid system
        split_square = VGroup(red_rect, green_rect).arrange(RIGHT, buff=0)
        self.place_in_area(split_square, 'C2', 'E5')
        
        # Transition from outline to filled regions
        self.play(
            FadeIn(split_square),
            factory_outline.animate.set_stroke(opacity=0.3),
            FadeOut(factory_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition highlight to the third line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # RESOLVING ISSUE 21: Integrate robot.svg asset for both labels
        # Glitch-Bot Label Group
        glitch_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg").set_color("#E74C3C").scale(0.15)
        glitch_text = Text("Glitch-Bot (20%)", font_size=18, color="#E74C3C")
        glitch_label = VGroup(glitch_icon, glitch_text).arrange(RIGHT, buff=0.1)
        
        # Normal-Bot Label Group
        normal_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg").set_color("#2ECC71").scale(0.15)
        normal_text = Text("Normal-Bot (80%)", font_size=18, color="#2ECC71")
        normal_label = VGroup(normal_icon, normal_text).arrange(RIGHT, buff=0.1)
        
        # RESOLVING ISSUE 29: Position glitch_label at B2 with scale 0.7
        self.place_at_grid(glitch_label, 'B2', scale_factor=0.7)
        
        # RESOLVING ISSUE 28: Center normal_label in area B3-B5 with scale 0.7
        self.place_in_area(normal_label, 'B3', 'B5', scale_factor=0.7)
        
        self.play(Write(glitch_label), Write(normal_label))
        
        # Pulse the 'Glitch-Bot' label to emphasize the initial belief (Prior)
        # Preserve Indicate color from storyboard/issue
        self.play(Indicate(glitch_label, color="#E74C3C"))
        self.wait(2)
