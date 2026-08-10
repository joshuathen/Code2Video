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
            "Transformers store facts as coordinates in high-dimensional space.",
            "Imagine a library where similar concepts cluster together.",
            "Vectors represent semantic meaning through spatial proximity."
        ]
        self.setup_layout("Prerequisite: The Concept of Vector Embeddings", lecture_lines)
        
        # Assets
        lib_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/library.svg")
        
        axes = Axes(x_range=[-1, 2, 1], y_range=[-1, 2, 1], x_length=4, y_length=4, axis_config={"include_tip": True})
        self.place_in_area(axes, "D2", "F5", scale_factor=0.6)

        cat_label = Text("Cat", font_size=24, color="#FF5733")
        dog_label = Text("Dog", font_size=24, color="#33FF57")
        self.place_at_grid(cat_label, "D5", scale_factor=0.8)
        self.place_at_grid(dog_label, "D2", scale_factor=0.8)
        
        # Vectors and Dot
        cat_vec = Vector([0.4, 0.1], color="#FF5733")
        dog_vec = Vector([0.1, 0.4], color="#33FF57")
        vector_group = VGroup(cat_vec, dog_vec)
        self.place_in_area(vector_group, "C3", "F6", scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF5733"))
        self.play(Create(axes), FadeIn(lib_icon.scale(0.5).next_to(axes, UP)))
        self.play(FadeIn(cat_label), GrowArrow(cat_vec))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#33FF57"))
        self.play(FadeIn(dog_label), GrowArrow(dog_vec))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#3357FF"))
        line = Line(axes.c2p(0.4, 0.1), axes.c2p(0.1, 0.4), color="#3357FF", stroke_width=4)
        lib_icon_2 = lib_icon.copy().scale(0.5).next_to(line, RIGHT)
        self.play(Create(line), FadeIn(lib_icon_2))
        self.wait(1)
