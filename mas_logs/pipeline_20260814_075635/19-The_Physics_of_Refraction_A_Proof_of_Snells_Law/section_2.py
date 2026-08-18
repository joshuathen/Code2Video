from manim import *

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
        self.setup_layout("Prerequisite Geometry: Defining the Interface", [
            "Consider two media with different indices.",
            "They are separated by a flat boundary.",
            "Measure angles relative to the normal."
        ])
        
        # Core Geometry
        boundary = Line(self.grid["D1"] + LEFT*1, self.grid["D6"] + RIGHT*1, color=BLUE)
        normal = DashedLine(self.grid["A4"], self.grid["F4"], color=WHITE)
        
        # Icons
        laser_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg")
        fiber_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fiber.svg")
        
        # Labels
        label_n1 = Text("n1", font_size=24, color=YELLOW)
        label_n2 = Text("n2", font_size=24, color=YELLOW)
        
        # Rays
        incident_ray = Line(self.grid["B2"], self.grid["D4"], color=RED)
        refracted_ray = Line(self.grid["D4"], self.grid["E6"], color=GREEN)
        
        # Angles
        theta1 = MathTex(r"\\theta_1", font_size=24, color=RED)
        theta2 = MathTex(r"\\theta_2", font_size=24, color=GREEN)
        
        # Positioning
        self.place_at_grid(label_n1, "B4")
        self.place_at_grid(label_n2, "E4")
        self.place_at_grid(laser_icon, "C3", scale_factor=0.3)
        self.place_at_grid(theta1, "C5", scale_factor=0.8)
        self.place_at_grid(theta2, "E5", scale_factor=0.8)
        self.place_at_grid(fiber_icon, "D5", scale_factor=0.3)
        
        # Animation sequence
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Write(label_n1), Write(label_n2))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        self.play(Create(boundary))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        self.play(
            Create(normal),
            FadeIn(laser_icon),
            Create(incident_ray),
            Create(refracted_ray),
            FadeIn(fiber_icon),
            Write(theta1),
            Write(theta2)
        )
        self.wait(2)
