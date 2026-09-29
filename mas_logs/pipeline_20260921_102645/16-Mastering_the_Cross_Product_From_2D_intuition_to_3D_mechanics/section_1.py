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
            "Vectors are directed segments in space.",
            "The Right-Hand Rule defines 3D orientation.",
            "The cross product creates a perpendicular vector."
        ]
        self.setup_layout("Conceptual Foundations", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Fade in two vectors in #FFFFFF. Label them #FF0000.
        v1 = Vector([1, 1, 0], color=WHITE)
        v2 = Vector([1.5, -0.5, 0], color=WHITE)
        l1 = Text("v1", color="#FF0000", font_size=20)
        l2 = Text("v2", color="#FF0000", font_size=20)
        
        # Asset integration
        hand = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hand.svg")
        
        self.place_at_grid(v1, "B3", scale_factor=0.9)
        self.place_at_grid(v2, "B4", scale_factor=0.9)
        self.place_at_grid(l1, "B1")
        self.place_at_grid(l2, "B5")
        self.place_at_grid(hand, "C4", scale_factor=0.5)
        
        self.play(FadeIn(v1), FadeIn(v2), FadeIn(l1), FadeIn(l2), FadeIn(hand))
        self.play(self.lecture[0].animate.set_color("#FF0000"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show dot product result as a scalar value. Label #00FF00.
        dot_res = Text("v1 · v2 = scalar", color="#00FF00", font_size=24)
        self.place_at_grid(dot_res, "C3", scale_factor=0.8)
        
        self.play(Write(dot_res))
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Illustrate vector projection visually with a dotted line #FFFF00.
        proj = DashedLine(v1.get_end(), v2.get_end(), color="#FFFF00")
        self.play(Create(proj))
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.wait(2)
