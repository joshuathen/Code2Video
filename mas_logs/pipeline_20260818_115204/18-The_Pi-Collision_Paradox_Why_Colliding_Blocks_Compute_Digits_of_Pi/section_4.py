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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "The circle radius represents total energy.",
            "Arc segments measure the circle's circumference.",
            "Angle segments correspond to Pi divisions.",
            "Mass ratios determine the arc lengths.",
            "The physics problem becomes pure geometry."
        ]
        self.setup_layout("The Arc and the Angle", lecture_lines)
        
        # Visual setup
        circle = Circle(radius=1.5, color=WHITE)
        # Applying grid fix from issue 30
        self.place_at_grid(circle, 'D4', scale_factor=0.6)
        
        # Using asset paths
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        
        center_point = Dot(color=WHITE)
        # Applying grid fix from issue 32
        self.place_at_grid(center_point, 'D4', scale_factor=0.3)
        label_c = Text("C", font_size=20).next_to(center_point, UP, buff=0.1)
        
        # Arc and labels
        arc = Arc(radius=0.9, start_angle=0, angle=PI/3, color=YELLOW)
        arc.move_to(circle.get_center())
        arc_label = Text("Arc", font_size=20)
        # Applying grid fix from issue 31
        self.place_at_grid(arc_label, 'D3', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(circle), FadeIn(center_point), Write(label_c))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # === Animation for Lecture Line 2 ===
        self.place_at_grid(protractor, 'B2', scale_factor=0.5)
        self.play(FadeIn(protractor))
        self.play(Create(arc))
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        
        # === Animation for Lecture Line 3 ===
        self.place_at_grid(compass, 'F5', scale_factor=0.5)
        pi_label = MathTex("\\pi", color="#FFFF00")
        pi_label.next_to(compass, RIGHT)
        self.play(FadeIn(compass), Write(pi_label))
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        # === Animation for Lecture Line 4 ===
        flash_point = Dot(arc.point_from_proportion(0.5), color="#FF4500", radius=0.1)
        self.play(Indicate(flash_point, color="#FF4500"))
        self.play(self.lecture[3].animate.set_color("#FF4500"))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#00FFFF"))
        self.wait(2)
