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
            "Think of vectors as familiar 2D arrows.",
            "Can these rules apply to other objects?",
            "We define a Vector Space with eight axioms.",
            "These rules transcend the objects' physical form.",
            "Objects simply need to satisfy these eight properties."
        ]
        self.setup_layout("From Concrete to Abstract", lecture_lines)
        
        # Initialize visuals
        # Use SVGMobject for asset
        apple_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/apple.svg"
        apples = VGroup(*[SVGMobject(apple_path, color="#FF6347") for _ in range(3)]).arrange(RIGHT, buff=0.2)
        self.place_at_grid(apples, 'B5', scale_factor=0.8)
        
        label_concrete = Text("Concrete", font_size=20, color=WHITE)
        self.place_at_grid(label_concrete, 'A4', scale_factor=1.0)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF6347")
        self.play(FadeIn(apples), Write(label_concrete))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#4682B4")
        dots = VGroup(*[Dot(color="#4682B4") for _ in range(3)]).arrange(RIGHT, buff=0.5)
        label_repr = Text("Representation", font_size=20, color=WHITE)
        self.place_at_grid(label_repr, 'C4', scale_factor=0.9)
        
        self.play(
            ReplacementTransform(apples, dots),
            FadeOut(label_concrete),
            FadeIn(label_repr)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#32CD32")
        # Reuse asset path for abstract
        abstract_objects = VGroup(*[SVGMobject(apple_path, color="#32CD32") for _ in range(3)]).arrange(RIGHT, buff=0.2)
        label_abstract = Text("Abstract", font_size=20, color=WHITE)
        self.place_at_grid(label_abstract, 'A4', scale_factor=1.0)
        
        self.play(
            ReplacementTransform(dots, abstract_objects),
            FadeOut(label_repr),
            FadeIn(label_abstract)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFD700")
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFD700")
        self.wait(2)
