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
        self.setup_layout("Introduction: What is a Vector?", [
            "Vectors are directed arrows in space.",
            "They represent both magnitude and direction.",
            "Unlike scalars, vectors have multiple components."
        ])
        
        # Initialize objects
        axes = Axes(x_range=[0, 6, 1], y_range=[0, 4, 1], axis_config={"include_numbers": False}).scale(0.5)
        # Applying layout improvements
        self.place_in_area(axes, 'B3', 'F6', scale_factor=0.6)
        
        vector = Arrow(start=ORIGIN, end=axes.c2p(3, 2), color=WHITE, buff=0)
        # Using grid-relative positioning as requested
        self.place_at_grid(vector, 'C4', scale_factor=0.7)
        
        label = MathTex(r"\vec{v}").next_to(vector.get_center(), UP)
        
        # Grid points visual (represented by a dot pattern for illustration)
        grid_points = VGroup(*[Dot(self.grid[f"{r}{c}"], radius=0.03) for r in ["B","C","D","E"] for c in ["2","3","4","5"]])
        self.place_in_area(grid_points, 'B2', 'E5', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(axes), Create(vector), Write(label), FadeIn(grid_points))
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        self.play(Indicate(vector, color="#FF5733"))
        self.lecture[1].set_color("#FF5733")
        
        # === Animation for Lecture Line 3 ===
        # Show magnitude and components visually
        mag_line = Line(ORIGIN, axes.c2p(3, 2), color="#33FF57")
        self.play(FadeIn(mag_line))
        self.lecture[2].set_color("#33FF57")
        
        self.wait(2)
