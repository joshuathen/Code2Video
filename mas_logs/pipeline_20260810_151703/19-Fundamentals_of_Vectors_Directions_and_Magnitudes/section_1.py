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
            "Scalars just show simple magnitude.",
            "Vectors represent both magnitude and direction.",
            "A cat's speed is a scalar.",
            "Velocity towards a bowl is a vector.",
            "Think of vectors as directional arrows."
        ]
        self.setup_layout("Fundamentals of Vectors: Directions and Magnitudes", lecture_lines)
        
        # Load Assets
        cat = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")
        bowl = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bowl.svg")
        
        # === Animation for Lecture Line 1 ===
        # Draw a small white point labeled "Scalar" (#FFFFFF) next to a cat
        dot = Dot(color=WHITE)
        scalar_text = Text("Scalar", font_size=18, color=WHITE)
        self.place_at_grid(dot, 'C5', scale_factor=0.9)
        self.place_at_grid(cat, 'C4', scale_factor=0.3)
        scalar_text.next_to(dot, RIGHT, buff=0.1)
        self.add(dot, scalar_text, cat)
        self.play(Create(dot), FadeIn(cat), Write(scalar_text))
        self.lecture[0].set_color(WHITE)

        # === Animation for Lecture Line 2 ===
        # Transform the point into a cyan arrow labeled "Vector" (#00FFFF).
        arrow = Arrow(start=ORIGIN, end=RIGHT*1.5, color="#00FFFF")
        vector_text = Text("Vector", font_size=18, color="#00FFFF")
        self.place_at_grid(arrow, 'C5')
        vector_text.next_to(arrow, UP, buff=0.1)
        self.play(ReplacementTransform(dot, arrow), ReplacementTransform(scalar_text, vector_text), FadeOut(cat))
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        # Display text "Vector has magnitude and direction" in light blue (#ADD8E6).
        label = Text("Vector has magnitude and direction", font_size=18, color="#ADD8E6")
        self.place_at_grid(label, 'D5', scale_factor=0.7)
        self.play(Write(label))
        self.lecture[2].set_color("#ADD8E6")

        # === Animation for Lecture Line 4 ===
        # Flash the cyan arrow to emphasize directionality.
        self.play(Indicate(arrow))
        self.lecture[3].set_color("#ADD8E6")

        # === Animation for Lecture Line 5 ===
        # Clear the scene while holding the "Vector" label and placing a bowl underneath it
        self.play(FadeOut(label), FadeOut(arrow))
        self.place_at_grid(bowl, 'D5', scale_factor=0.5)
        self.place_at_grid(vector_text, 'C5')
        self.play(FadeIn(bowl))
        self.lecture[4].set_color("#00FFFF")
