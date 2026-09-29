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
        self.setup_layout("The 3D Cross Product: Definition and Right-Hand Rule", [
            "In 3D, cross product produces a normal vector.",
            "Use the Right-Hand Rule to find its direction.",
            "Curl your fingers from vector A to vector B."
        ])
        
        # Elements - using 3DThreeDScene approach with restricted camera
        # To mimic 3D but stay in 2D projection as per instructions
        v = Arrow(ORIGIN, RIGHT * 1.5, color="#00FF00")
        w = Arrow(ORIGIN, UP * 1.5, color="#0000FF")
        result = Arrow(ORIGIN, np.array([0.5, 0.5, 1.5]), color="#FFFF00")
        
        vectors = VGroup(v, w, result)
        self.place_in_area(vectors, "C3", "E5", scale_factor=0.6)

        # Hand SVG
        hand = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hand.svg")
        self.place_at_grid(hand, "D5", scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        self.play(Create(v), Create(w))
        self.wait(0.5)
        self.play(Create(result))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#0000FF"))
        self.play(FadeIn(hand))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        # Animate hand gesture
        self.play(hand.animate.rotate(PI/4).shift(UP*0.2))
        self.wait(2)
