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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Core Constraints: The Axiomatic DNA", [
            "Spaces must pass the stress test.",
            "Closure requires staying within the set.",
            "Sum of vectors must be inside."
        ])
        
        # Polynomial Set Container
        poly_container = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/container.svg")
        self.place_in_area(poly_container, "B2", "D4", scale_factor=0.6)
        
        # Gatekeeper
        gatekeeper = Dot(radius=0.3, color="#FF5733")
        self.place_at_grid(gatekeeper, "B6", scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(poly_container), GrowFromCenter(gatekeeper))
        self.lecture[0].set_color("#FF5733")

        # === Animation for Lecture Line 2 ===
        # Valid sum operation
        vec_sum = VGroup(Dot(color="#33FF57"), Dot(color="#33FF57")).arrange(RIGHT)
        self.place_at_grid(vec_sum, "C3")
        self.play(FadeIn(vec_sum))
        self.play(poly_container.animate.set_color("#33FF57"), gatekeeper.animate.set_color("#33FF57"))
        self.lecture[1].set_color("#33FF57")

        # === Animation for Lecture Line 3 ===
        # Invalid sum operation (outside box)
        invalid_sum = VGroup(Dot(color="#FF3333"), Dot(color="#FF3333")).arrange(RIGHT)
        self.place_at_grid(invalid_sum, "E1", scale_factor=0.8)
        self.play(FadeIn(invalid_sum))
        self.play(poly_container.animate.set_color("#FF3333"), gatekeeper.animate.set_color("#FF3333"))
        self.lecture[2].set_color("#FF3333")
