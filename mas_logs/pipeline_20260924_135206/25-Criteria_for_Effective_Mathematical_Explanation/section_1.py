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
        lecture_lines = [
            "Effective explanations bridge logic to learner schemas.",
            "We use the 3C framework for clarity.",
            "Clarity, Connection, and Completeness anchor concepts."
        ]
        self.setup_layout("Introduction: The Goal of Clarity", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00")
        
        # Fade in a central lightbulb graphic icon in #FFFF00
        # Loading SVG Asset
        self.lightbulb = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lightbulb.svg")
        self.lightbulb.set_color("#FFFF00")
        
        # Add label
        self.label = Text("Clarity", font_size=24, color="#FFFFFF")
        
        # Position using fixes for issue 20, 21, 22
        self.place_at_grid(self.lightbulb, 'C5', scale_factor=0.5)
        self.place_at_grid(self.label, 'C6', scale_factor=1.0)
        
        self.play(FadeIn(self.lightbulb), Write(self.label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FF00FF")
        
        # Pulse the lightbulb graphic to emphasize the 'Clarity' goal.
        self.play(
            Indicate(self.lightbulb, color="#FFFFFF", scale_factor=1.2),
            run_time=2
        )
        self.wait(1)
