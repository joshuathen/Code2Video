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
        self.setup_layout("The 'Why': Explicit vs. Implicit", [
            "Explicit functions express y in terms of x.",
            "Implicit equations relate x and y together.",
            "Circles break the vertical line test."
        ])
        
        # === Animation for Lecture Line 1 ===
        explicit_label = Text("Explicit Function y = f(x)", font_size=36, color=WHITE)
        self.place_at_grid(explicit_label, 'B3', scale_factor=0.7)
        self.play(Write(explicit_label))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        implicit_label = Text("Implicit F(x, y) = 0", font_size=36, color=WHITE)
        self.place_at_grid(implicit_label, 'D3', scale_factor=0.7)
        # Using a VGroup to combine for the requested area-based placement
        formula_group = VGroup(explicit_label, implicit_label)
        self.place_in_area(formula_group, 'B2', 'E5', scale_factor=0.65)
        
        self.play(ReplacementTransform(explicit_label, implicit_label))
        self.lecture[1].set_color(ORANGE)
        self.play(FadeOut(implicit_label))

        # === Animation for Lecture Line 3 ===
        # Using SVG asset
        circle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg", color="#00FFFF")
        self.place_at_grid(circle, 'C5', scale_factor=0.6)
        
        # Ensure the circle is fully rendered/processed so points exist before calling point_from_proportion
        self.play(Create(circle))
        
        # If the loaded SVG has no path points, use Circle() as a fallback to ensure points exist
        circle_path = circle if circle.has_points() else Circle(radius=0.5).move_to(circle.get_center())
        
        dot1 = Dot(color=RED).move_to(circle_path.point_from_proportion(0)) # (1,0)
        dot2 = Dot(color=RED).move_to(circle_path.point_from_proportion(0.25)) # (0,1)
        
        self.play(FadeIn(dot1), FadeIn(dot2))
        self.lecture[2].set_color(BLUE)
        self.wait(1)
