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
        self.setup_layout("Summary and Intuition", [
            "Complex numbers solve discrete counting.",
            "Algebraic evaluation replaces manual analysis.",
            "Vectors cancel to leave pure sums."
        ])
        
        # Load asset
        asset_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        
        # Geometry: Roots of Unity / Cancellation
        center = self.grid["B3"] # Adjusted away from C3 to avoid overlap
        circle = Circle(radius=1.5, color=BLUE).move_to(center)
        roots = VGroup(*[
            Dot(color=YELLOW).move_to(center + 1.5 * RIGHT * np.cos(i * 2 * PI / 5) + 1.5 * UP * np.sin(i * 2 * PI / 5))
            for i in range(5)
        ])
        vectors = VGroup(*[
            Arrow(start=center, end=root.get_center(), color=RED, buff=0)
            for root in roots
        ])
        
        animation_group = VGroup(circle, roots, vectors, asset_icon)
        self.place_at_grid(animation_group, 'B4', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(Create(circle), FadeIn(roots), FadeIn(asset_icon))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.play(Create(vectors))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(Rotate(vectors, angle=PI/2, about_point=center), run_time=2)
        self.play(FadeOut(animation_group))
        
        efficiency_text = Text("Efficiency: O(n) vs O(2^n)", font_size=32, color=ORANGE)
        self.place_at_grid(efficiency_text, 'E2', scale_factor=0.7)
        self.play(FadeIn(efficiency_text))
        
        # Handle asset scaling as requested by critics
        visual_elements = VGroup(efficiency_text)
        self.place_in_area(visual_elements, 'D3', 'F5', scale_factor=0.5)
        
        self.wait(2)
