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
        self.setup_layout("Synthesis: Multi-Head Attention", [
            "Single spotlight cannot capture everything.",
            "Multi-Head Attention runs parallel processes.",
            "Different heads capture different linguistic aspects."
        ])
        
        # Colors for lecture lines
        colors = ["#FF7F50", "#87CEFA", "#90EE90"]
        
        # Assets
        spotlight_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/spotlight.svg"

        # === Animation for Lecture Line 1 ===
        # Visualize single spotlight on a text
        text_target = Text("The cat sat on the mat.", font_size=30)
        self.place_in_area(text_target, 'A1', 'A6', scale_factor=0.6)
        
        spotlight = SVGMobject(spotlight_path)
        spotlight.set_color(YELLOW)
        spotlight.scale(0.5)
        spotlight.move_to(text_target.get_center())
        
        self.play(FadeIn(text_target), Create(spotlight))
        self.lecture[0].set_color(colors[0])
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show multiple parallel heads
        # Using SVG for spotlights as per instructions
        heads = VGroup(*[SVGMobject(spotlight_path).scale(0.3).set_color(c) for c in [RED, BLUE, GREEN]])
        heads.arrange(RIGHT, buff=0.2)
        # Positioned per constraint
        self.place_at_grid(heads, 'D3', scale_factor=0.9)
        
        self.play(FadeOut(spotlight), FadeIn(heads))
        self.lecture[1].set_color(colors[1])
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Different heads focusing on different patterns
        arrows = VGroup(*[Arrow(start=h.get_top(), end=text_target.get_bottom(), color=h.get_color(), buff=0.1) for h in heads])
        
        # Combine results into a single enriched representation using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/spotlight.svg]
        final_spotlight = SVGMobject(spotlight_path).scale(0.6).set_color(WHITE)
        self.place_at_grid(final_spotlight, 'E3')
        
        self.play(Create(arrows))
        self.play(FadeOut(heads), FadeOut(arrows), FadeIn(final_spotlight))
        self.lecture[2].set_color(colors[2])
        self.wait(2)
