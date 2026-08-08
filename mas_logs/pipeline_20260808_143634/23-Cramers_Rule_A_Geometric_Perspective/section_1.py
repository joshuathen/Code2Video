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
        self.setup_layout("Prerequisites: Vectors as Areas", [
            "Determinants represent the area of a parallelogram.",
            "Vectors span a space with signed area.",
            "If collinear, area becomes zero."
        ])
        
        # Setup axes
        axes = Axes(
            x_range=[-1, 3, 1],
            y_range=[-1, 3, 1],
            axis_config={"color": WHITE, "include_numbers": False},
        )
        # Using fix for issue 19
        self.place_in_area(axes, 'C2', 'F6', scale_factor=0.5)

        # Create vectors (using fixed axes)
        origin = axes.c2p(0, 0)
        v1 = Vector(axes.c2p(2, 0) - origin, color=BLUE).shift(origin)
        v2 = Vector(axes.c2p(0, 2) - origin, color=RED).shift(origin)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700") # Gold
        self.play(Create(axes), Create(v1), Create(v2))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00") # Green
        
        parallelogram = Polygon(
            origin, axes.c2p(2, 0), axes.c2p(2, 2), axes.c2p(0, 2),
            fill_color=PURPLE, fill_opacity=0.3, stroke_width=2, stroke_color=PURPLE
        )
        self.play(FadeIn(parallelogram))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF6347") # Tomato
        
        v3 = Vector(axes.c2p(2, 1) - origin, color=RED).shift(origin)
        # Showing collinear case by setting v2 to v1
        self.play(
            FadeOut(parallelogram),
            v2.animate.become(Vector(axes.c2p(1, 0) - origin, color=RED).shift(origin))
        )
        self.wait(1)
