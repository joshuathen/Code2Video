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

class Section6Scene(TeachingScene):
    def construct(self):
        # Define lecture lines
        lecture_lines = [
            'Order matters because matrix multiplication is non-commutative.',
            'Shearing then rotating yields a different final result.',
            '[Asset: Robo-Cat] reaches a unique position for each order.',
            'This proves that A-B does not equal B-A.',
            'Sequential actions in space are order-dependent.'
        ]
        
        self.setup_layout("Summary & Order Sensitivity", lecture_lines)
        
        # Colors for highlighting
        colors = [YELLOW, TEAL, GREEN, RED, PURPLE]
        
        # Matrices
        # A = Shear, B = Rotate
        shear_matrix = [[1, 1], [0, 1]]
        rotate_matrix = [[0, -1], [1, 0]]
        
        # Grids and Labels
        plane_left = NumberPlane(x_range=[-2, 2, 1], y_range=[-2, 2, 1], 
                                background_line_style={"stroke_opacity": 0.4})
        plane_right = NumberPlane(x_range=[-2, 2, 1], y_range=[-2, 2, 1], 
                                 background_line_style={"stroke_opacity": 0.4})
        
        self.place_in_area(plane_left, 'A1', 'C3', scale_factor=0.4)
        self.place_in_area(plane_right, 'A4', 'C6', scale_factor=0.4)
        
        label_ba = Text("BA: Shear then Rotate", font_size=16, color=WHITE)
        label_ab = Text("AB: Rotate then Shear", font_size=16, color=WHITE)
        
        # Issue 50 & 51 Fix: Use place_in_area for multi-word labels
        self.place_in_area(label_ba, 'D1', 'D3', scale_factor=0.6)
        self.place_in_area(label_ab, 'D4', 'D6', scale_factor=0.6)

        # Issue 33 Fix: [Asset: Robo-Cat] integration
        cat_asset_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/cat.png"
        cat_ba = ImageMobject(cat_asset_path)
        cat_ab = ImageMobject(cat_asset_path)

        # Positioning cats in the center of the same area as planes
        self.place_in_area(cat_ba, 'A1', 'C3', scale_factor=0.3)
        self.place_in_area(cat_ab, 'A4', 'C6', scale_factor=0.3)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(colors[0]))
        self.play(
            FadeIn(plane_left), FadeIn(plane_right),
            FadeIn(label_ba), FadeIn(label_ab),
            FadeIn(cat_ba), FadeIn(cat_ab)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(colors[1]))
        # Sequence BA: First step is Shear (A)
        # Sequence AB: First step is Rotate (B)
        self.play(
            cat_ba.animate.apply_matrix(shear_matrix),
            cat_ab.animate.apply_matrix(rotate_matrix),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(colors[2]))
        # Sequence BA: Second step is Rotate (B)
        # Sequence AB: Second step is Shear (A)
        self.play(
            cat_ba.animate.apply_matrix(rotate_matrix),
            cat_ab.animate.apply_matrix(shear_matrix),
            run_time=2
        )
        
        # Highlight final positions
        box_ba = SurroundingRectangle(cat_ba, color=YELLOW, buff=0.1)
        box_ab = SurroundingRectangle(cat_ab, color=YELLOW, buff=0.1)
        self.play(Create(box_ba), Create(box_ab))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(colors[3]))
        # Display 'A * B is not equal to B * A' in bold white (#FFFFFF)
        not_equal_text = Text("A · B ≠ B · A", color=WHITE, font_size=36)
        self.place_in_area(not_equal_text, 'E1', 'E6', scale_factor=1.0)
        self.play(Write(not_equal_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(colors[4]))
        conclusion_text = Text("Order Matters in Composition", font_size=32, color=WHITE)
        # Issue 49 Fix: Moved conclusion_text to F1-F6 to avoid overlap
        self.place_in_area(conclusion_text, 'F1', 'F6', scale_factor=0.8)
        
        # Fade out everything except the final conclusion text and title
        to_remove = [
            plane_left, plane_right, label_ba, label_ab, 
            cat_ba, cat_ab, box_ba, box_ab, not_equal_text, self.lecture
        ]
        
        self.play(
            *[FadeOut(m) for m in to_remove],
            Write(conclusion_text)
        )
        self.wait(2)
