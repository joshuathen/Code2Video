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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: Angular Sweep", [
            "Imagine a line rotating around a pivot point.",
            "The Windmill problem uses this rotation strategy.",
            "A line strikes points in a set."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Create pivot point P
        pivot = Dot(color="#FFFFFF")
        pivot_label = Text("P", font_size=24, color="#FFFFFF").next_to(pivot, UP, buff=0.1)
        pivot_group = VGroup(pivot, pivot_label)
        # Apply fix for Issue 22
        self.place_at_grid(pivot_group, "C4", scale_factor=1.0)
        self.play(FadeIn(pivot_group))
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF4500")
        # Define set of points
        points = VGroup(*[Dot(self.grid[pos], color="#FF4500") for pos in ["B2", "B4", "D2", "D4", "E3"]])
        
        # Use asset as per Issue 16
        windmill_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/windmill.svg")
        self.place_in_area(windmill_icon, "C3", "C3", scale_factor=0.5)
        
        self.play(FadeIn(points), FadeIn(windmill_icon))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        # Creating a rotating line based on the asset and points
        line = Line(start=self.grid["C4"] + LEFT*2, end=self.grid["C4"] + RIGHT*2, color="#FF4500")
        
        self.play(Create(line))
        self.play(Rotate(line, angle=PI/2, about_point=self.grid["C4"], run_time=2))
        self.wait(1)
