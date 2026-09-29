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
            "Words convert into high-dimensional numerical vectors.",
            "Similar words cluster closely in semantic space.",
            "'Cat' and 'Dog' appear nearby.",
            "'Airplane' remains far from animals."
        ]
        self.setup_layout("Prerequisite: The Concept of Word Embeddings", lecture_lines)
        
        # Prepare semantic points
        # Using assets for Airplane and Cat
        airplane_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/airplane.svg")
        cat_asset = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")
        dog = Dot(color=BLUE).scale(1.5)
        
        cat_label = Text("Cat", font_size=20)
        dog_label = Text("Dog", font_size=20)
        airplane_label = Text("Airplane", font_size=20)

        # === Animation for Lecture Line 1 ===
        # Words convert into high-dimensional numerical vectors.
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        vec_display = Tex(r"$\vec{v}_{word} = [0.1, -0.5, 0.8, \dots]$", font_size=28)
        self.place_in_area(vec_display, 'A3', 'A6', scale_factor=0.9)
        self.play(FadeIn(vec_display))
        self.place_at_grid(airplane_asset, 'B3', scale_factor=0.4)
        self.play(FadeIn(airplane_asset))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Similar words cluster closely in semantic space.
        self.play(self.lecture[1].animate.set_color("#808080"))
        cluster_area = Rectangle(width=3, height=3, color=GRAY, stroke_opacity=0.5)
        self.place_in_area(cluster_area, 'C3', 'E4', scale_factor=0.8)
        self.play(FadeIn(cluster_area))
        
        # === Animation for Lecture Line 3 ===
        # 'Cat' and 'Dog' appear nearby.
        self.play(self.lecture[2].animate.set_color("#FF0000"))
        self.place_at_grid(cat_asset, 'C3', scale_factor=0.15)
        self.place_at_grid(dog, 'D4')
        self.place_at_grid(cat_label, 'C2', scale_factor=0.7)
        self.place_at_grid(dog_label, 'D5', scale_factor=0.7)
        self.play(FadeIn(cat_asset), FadeIn(dog), FadeIn(cat_label), FadeIn(dog_label))
        
        distance_line = Line(cat_asset.get_center(), dog.get_center(), color=RED)
        self.play(Create(distance_line))
        self.wait(1)
        self.play(FadeOut(distance_line))

        # === Animation for Lecture Line 4 ===
        # 'Airplane' remains far from animals.
        self.play(self.lecture[3].animate.set_color("#FF0000"))
        self.place_at_grid(airplane_label, 'F3', scale_factor=0.7)
        self.play(FadeIn(airplane_label))
        self.wait(2)
