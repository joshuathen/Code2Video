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
            "Matrices are functions transforming inputs into outputs.",
            "A 2D vector gets mapped by a 2x2 matrix.",
            "Transformation means rotation, scaling, or shearing."
        ]
        self.setup_layout("Prerequisite Refresher: The Mapping Perspective", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Show a vector space R^2. (Color: #FFFFFF)
        axes = Axes(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_tip": True}).scale(0.5)
        self.place_in_area(axes, 'C1', 'E2', scale_factor=0.6)
        vector = Vector([1, 2], color=WHITE)
        vector.move_to(axes.c2p(0.5, 1)) # Anchor vector
        self.add(axes, vector)
        self.lecture[0].set_color("#FFFFFF")
        self.play(Create(axes), GrowArrow(vector))

        # === Animation for Lecture Line 2 ===
        # Represent the mapping as an arrow from R^n to R^m. (Color: #FF8080)
        self.lecture[1].set_color("#FF8080")
        target_axes = Axes(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_tip": True}).scale(0.5)
        self.place_in_area(target_axes, 'C4', 'E5', scale_factor=0.6)
        mapping_arrow = Arrow(start=LEFT*0.5, end=RIGHT*0.5, color="#FF8080")
        
        T_label = Text("T", color="#FF8080")
        self.place_at_grid(T_label, 'B3', scale_factor=0.7)
        mapping_arrow.next_to(T_label, DOWN, buff=0.1)
        
        self.play(Create(target_axes), FadeIn(mapping_arrow), Write(T_label))

        # === Animation for Lecture Line 3 ===
        # Visualize the domain R^n and codomain R^m separately. (Color: #80FF80)
        self.lecture[2].set_color("#80FF80")
        new_vector = Vector([2, -1], color="#80FF80")
        new_vector.move_to(target_axes.c2p(1, -0.5))
        
        self.play(Transform(vector.copy(), new_vector), run_time=2)
        self.wait(1)
