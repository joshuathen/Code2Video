from manim import *
import numpy as np

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

class Section7ApplicationsScene(TeachingScene):
    def construct(self):
        # Setup layout
        lines = [
            "Topology helps scientists understand complex DNA knotting.",
            "It optimizes data analysis and electronic circuit design.",
            "These principles solve real-world problems through shape analysis."
        ]
        self.setup_layout("Modern Applications", lines)
        
        # DNA Color and Asset
        dna_color = "#00FA9A"
        dna_asset_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/dna.svg"
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Use DNA asset [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/dna.svg]
        loop1 = SVGMobject(dna_asset_path).set_color(dna_color)
        loop2 = SVGMobject(dna_asset_path).set_color(dna_color)
        
        # Following Critic fixes (Issue 51, 52): Initial placement at C2 and C5
        self.place_at_grid(loop1, "C2", scale_factor=0.8)
        self.place_at_grid(loop2, "C5", scale_factor=0.8)
        
        # Grid positions for animation
        pos_c2 = self.grid["C2"]
        pos_c5 = self.grid["C5"]
        pos_c3 = self.grid["C3"]
        pos_c4 = self.grid["C4"]
        
        # Initially separate, then move to intertwined state for line 1
        self.play(FadeIn(loop1), FadeIn(loop2))
        self.play(
            loop1.animate.move_to(pos_c3).rotate(30*DEGREES),
            loop2.animate.move_to(pos_c4).rotate(-30*DEGREES),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Animate unknotting: loops slide back to C2 and C5 (Issue 53)
        self.play(
            loop1.animate.move_to(pos_c2).rotate(-30*DEGREES),
            loop2.animate.move_to(pos_c5).rotate(30*DEGREES),
            run_time=2,
            rate_func=smooth
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Label the final circles 'Unknotted' (#FFFFFF)
        # Position labels at D2 and D5 to align with loops at C2 and C5
        label1 = Text("Unknotted", font_size=20, color=WHITE)
        label2 = Text("Unknotted", font_size=20, color=WHITE)
        
        self.place_at_grid(label1, "D2", scale_factor=1.0)
        self.place_at_grid(label2, "D5", scale_factor=1.0)
        
        self.play(Write(label1), Write(label2))
        self.wait(2)
        
        # Cleanup colors
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
