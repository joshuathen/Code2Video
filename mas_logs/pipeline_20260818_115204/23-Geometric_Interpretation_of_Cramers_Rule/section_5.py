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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary and Geometric Constraint", [
            "Singular systems have a zero determinant.", 
            "Columns collapse into a single line.", 
            "The target vector becomes unreachable."
        ])
        
        # Setup geometric elements
        v1 = Vector([1.5, 0.5], color=RED)
        v2 = Vector([3.0, 1.0], color=RED)
        geo_group = VGroup(v1, v2)
        
        target_b = Dot(color=BLUE)
        b_label = Text("b", font_size=20, color=BLUE)

        # === Animation for Lecture Line 1 ===
        # Use place_in_area for geometric elements to avoid overlap
        self.place_in_area(geo_group, 'D2', 'F6', scale_factor=0.9)
        self.play(FadeIn(v1), FadeIn(v2))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        # Animating collapse to a single line
        self.play(v2.animate.put_start_and_end_on(ORIGIN, [1.5, 0.5, 0]))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        # Using place_at_grid for point b as requested
        self.place_at_grid(target_b, 'F5', scale_factor=0.8)
        b_label.next_to(target_b, RIGHT)
        self.play(FadeIn(target_b), Write(b_label))
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
