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
        lecture_lines = [
            "MLPs store facts in diffuse patterns.",
            "This allows for flexible model generalization.",
            "Precise fact-checking remains inherently difficult."
        ]
        self.setup_layout("Synthesis & Limitations", lecture_lines)
        
        # Elements
        # Using SVG placeholder as per requirement
        cloud_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        cloud = VGroup(*[Dot(color=BLUE_C).move_to(self.grid["B3"] + np.random.uniform(-0.5, 0.5, 3)) for _ in range(20)])
        
        # Generalization map shifted to B2
        generalization_map = VGroup(*[Square(side_length=0.3, color=GREEN_B, fill_opacity=0.5) for _ in range(9)]).arrange_in_grid(3, 3)
        self.place_at_grid(generalization_map, "B2", scale_factor=0.7)
        
        # Fact-check box shifted to E2
        fact_check_box = RoundedRectangle(corner_radius=0.1, height=1.5, width=2.0, color=RED_C)
        self.place_at_grid(fact_check_box, "E2", scale_factor=0.8)
        cross = VGroup(Line(UP+LEFT, DOWN+RIGHT), Line(UP+RIGHT, DOWN+LEFT)).set_color(RED_E).scale(0.5).move_to(fact_check_box)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(cloud), FadeIn(cloud_icon.move_to(self.grid["B3"])))
        self.lecture[0].set_color(BLUE_C)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(generalization_map))
        self.lecture[1].set_color(GREEN_B)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        summary_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        self.play(Create(fact_check_box), Create(cross), FadeIn(summary_icon.move_to(self.grid["E5"])))
        self.lecture[2].set_color(RED_C)
        self.wait(2)
