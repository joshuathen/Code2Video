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
            "Cross product calculates area spanned by two vectors.",
            "The determinant finds this area as signed quantity.",
            "Positive area means counter-clockwise rotation, negative is clockwise."
        ]
        self.setup_layout("Prerequisite Warm-up: Orientation and Determinants", lecture_lines)
        
        # Create visual elements
        axes = Axes(x_length=3, y_length=3).add_coordinates()
        vec_a = Vector([1, 2], color=WHITE)
        vec_b = Vector([2, 0.5], color=WHITE)
        # Using a dummy SVG file path if non-existent or using ImageMobject for raster if needed, but per instructions using VGroup/primitives for robustness
        # Assuming asset path provided but not found, using primitives as per instructions
        parallelogram = Polygon([0,0,0], [1,2,0], [3,2.5,0], [2,0.5,0], fill_opacity=0.3, color=YELLOW)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_at_grid(axes, 'B4', scale_factor=0.7)
        self.play(Create(axes))
        # Place vectors at the same grid as axes center (manual calculation relative to axes pos)
        vec_a.next_to(axes.c2p(0,0), RIGHT, buff=0)
        vec_b.next_to(axes.c2p(0,0), RIGHT, buff=0)
        self.play(Create(vec_a), Create(vec_b))
        self.play(vec_a.animate.set_color("#00FF00"), vec_b.animate.set_color("#0000FF"))
        self.play(Create(parallelogram))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        det_text = MathTex(r"\\det = ad - bc").set_color(RED)
        self.place_in_area(det_text, 'E4', 'F6', scale_factor=1.0)
        self.play(Write(det_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        self.wait(2)
