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

class Section7Scene(TeachingScene):
    def construct(self):
        # Initializing layout
        lecture_lines = [
            "Determinants scale, flip, or collapse our mathematical world.",
            "Graphics engines use them to render shadows and reflections.",
            "It is the fundamental scaling factor of linear algebra."
        ]
        self.setup_layout("Summary & Real-world Intuition", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Three side-by-side visualizations with scaling, flipping, and collapsing
        
        # Asset Paths
        shadow_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/shadows.svg"
        
        # 1. Scaled (Yellow)
        grid1 = NumberPlane(x_range=[-2, 2, 1], y_range=[-2, 2, 1], x_length=2, y_length=2, 
                          background_line_style={"stroke_opacity": 0.3}, axis_config={"include_tip": False})
        self.place_in_area(grid1, 'A1', 'C2', scale_factor=0.6)
        
        scaled_square = Square(side_length=1.0, color=YELLOW, fill_opacity=0.4)
        self.place_in_area(scaled_square, 'A1', 'C2', scale_factor=0.9) # Issue 50 fix
        
        shadow1 = SVGMobject(shadow_path, color=GRAY_E, fill_opacity=0.3)
        self.place_in_area(shadow1, 'A1', 'C2', scale_factor=0.7).shift(DOWN*0.15 + RIGHT*0.1)
        
        # 2. Flipped (Purple)
        grid2 = grid1.copy()
        self.place_in_area(grid2, 'A3', 'C4', scale_factor=0.6)
        
        flipped_square = Square(side_length=1.0, color=PURPLE, fill_opacity=0.4).stretch(-1, 0)
        self.place_in_area(flipped_square, 'A3', 'C4', scale_factor=0.9)
        
        shadow2 = shadow1.copy()
        self.place_in_area(shadow2, 'A3', 'C4', scale_factor=0.7).shift(DOWN*0.15 + RIGHT*0.1)
        
        # 3. Collapsed (Red)
        grid3 = grid1.copy()
        self.place_in_area(grid3, 'A5', 'C6', scale_factor=0.6)
        
        collapse_line = Line(LEFT, RIGHT, color=RED, stroke_width=6)
        self.place_at_grid(collapse_line, 'C6', scale_factor=1.0) # Issue 52 fix
        
        shadow3 = shadow1.copy()
        self.place_at_grid(shadow3, 'C6', scale_factor=0.7).shift(DOWN*0.15 + RIGHT*0.1)

        self.play(
            FadeIn(grid1, grid2, grid3),
            FadeIn(shadow1, shadow2, shadow3),
            Create(scaled_square), Create(flipped_square), Create(collapse_line)
        )
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Adding labels and transformation references
        ref_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/ref.svg"
        
        label1 = Text("Det > 0", font_size=20, color=YELLOW)
        label2 = Text("Det < 0", font_size=20, color=PURPLE)
        label3 = Text("Det = 0", font_size=20, color=RED)
        
        ref1 = SVGMobject(ref_path).set_color(WHITE)
        ref2 = ref1.copy()
        ref3 = ref1.copy()

        # Anchoring labels as per issues
        self.place_at_grid(label1, 'E2', scale_factor=0.8) # Issue 50 fix
        self.place_at_grid(label2, 'E4', scale_factor=0.6) # Issue 51 fix
        self.place_at_grid(label3, 'E6', scale_factor=0.8) # Issue 52 fix
        
        self.place_at_grid(ref1, 'F2', scale_factor=0.4)
        self.place_at_grid(ref2, 'F4', scale_factor=0.4)
        self.place_at_grid(ref3, 'F6', scale_factor=0.4)

        self.play(
            Write(label1), Write(label2), Write(label3),
            FadeIn(ref1, ref2, ref3)
        )
        self.play(self.lecture[1].animate.set_color(PURPLE))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Final Summary: Morphing everything into the Magic Scaling Factor
        final_summary = Text("The Magic Scaling Factor", font_size=32, color="#FFFF00")
        self.place_in_area(final_summary, 'B1', 'E6', scale_factor=1.1)

        self.play(
            ReplacementTransform(VGroup(label1, label2, label3), final_summary),
            FadeOut(VGroup(grid1, grid2, grid3, shadow1, shadow2, shadow3, scaled_square, flipped_square, collapse_line, ref1, ref2, ref3))
        )
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.wait(2)
