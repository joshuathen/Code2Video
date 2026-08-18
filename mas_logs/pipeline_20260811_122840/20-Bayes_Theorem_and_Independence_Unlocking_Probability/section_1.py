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
        lecture_lines = ["Conditional probability focuses on a specific subset.", "We shrink our world to circle B.", "Only the overlap matters now."]
        self.setup_layout("Prerequisite Review: Conditional Probability", lecture_lines)
        
        # Elements
        sample_space = Rectangle(width=4, height=4, color=WHITE, fill_opacity=0.2)
        circle_a = Circle(radius=0.8, color="#44AAFF", fill_opacity=0.6)
        circle_b = Circle(radius=0.8, color="#44AAFF", fill_opacity=0.6)
        
        # Intersection icon
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg]
        intersection_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg", color="#FF5555")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#44AAFF"))
        self.place_in_area(sample_space, 'C3', 'E5', scale_factor=0.6)
        self.place_in_area(circle_a, 'C3', 'D4', scale_factor=0.7)
        self.place_in_area(circle_b, 'C4', 'D5', scale_factor=0.7)
        self.play(FadeIn(sample_space), Create(circle_a), Create(circle_b))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFCC00"))
        
        # "Shrink the world"
        new_sample_space = Rectangle(width=circle_b.width, height=circle_b.height, color="#FFCC00", fill_opacity=0.3)
        new_sample_space.move_to(circle_b.get_center())
        
        self.play(
            FadeOut(sample_space),
            FadeOut(circle_a),
            Transform(circle_b, new_sample_space)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF5555"))
        self.place_at_grid(intersection_icon, 'C5', scale_factor=0.5)
        self.play(FadeIn(intersection_icon))
        self.wait(2)
