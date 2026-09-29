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
        self.setup_layout("2D 'Cross Product' as Oriented Area", [
            "The 2D cross product measures oriented area.",
            "A parallelogram forms between two vectors.",
            "Counter-clockwise sequences yield positive results."
        ])
        
        # Setup objects
        v1 = Vector([1, 0], color=WHITE)
        v2 = Vector([0, 1], color=WHITE)
        label_v1 = MathTex(r"\vec{v}_1", color=RED).next_to(v1.get_end(), RIGHT)
        label_v2 = MathTex(r"\vec{v}_2", color=RED).next_to(v2.get_end(), UP)
        
        parallelogram = Polygon(
            ORIGIN, [1, 0, 0], [1, 1, 0], [0, 1, 0],
            fill_opacity=0.5, color=PURPLE
        )
        
        det_text = MathTex(r"\det(M) = x_1y_2 - x_2y_1", color=TEAL)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(TEAL))
        # Updated line 68 as requested
        self.place_at_grid(v1, "C3", scale_factor=1.0)
        self.place_at_grid(v2, "C3", scale_factor=1.0)
        self.add(v1, v2, label_v1, label_v2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(PURPLE))
        # Updated line 75 as requested
        self.place_in_area(parallelogram, 'B3', 'D5', scale_factor=1.2)
        self.play(Create(parallelogram))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        # Updated line 81 as requested
        self.place_at_grid(det_text, 'D4', scale_factor=0.9)
        self.play(Write(det_text))
        self.wait(2)
