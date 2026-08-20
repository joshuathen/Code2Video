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
        self.setup_layout("Prerequisites: Vectors as Area Operators", [
            "Determinants represent the area of a parallelogram.", 
            "Vectors define the parallelogram's sides.", 
            "Zero determinant means area collapses to zero."
        ])
        
        # Create axes for visualization
        axes = Axes(x_range=[-1, 4], y_range=[-1, 4], x_length=4, y_length=4, axis_config={"include_tip": True})
        self.place_in_area(axes, 'A2', 'C5', scale_factor=0.65)
        self.add(axes)
        
        # Vectors
        v1_obj = Arrow(axes.c2p(0, 0), axes.c2p(2, 1), color="#FF00FF", buff=0)
        v2_obj = Arrow(axes.c2p(0, 0), axes.c2p(1, 2), color="#00FFFF", buff=0)
        b_obj = Arrow(axes.c2p(0, 0), axes.c2p(3, 3), color="#FFFF00", buff=0)
        
        # Labels
        v1_label = MathTex("v_1", color="#FF00FF")
        v2_label = MathTex("v_2", color="#00FFFF")
        b_label = MathTex("b", color="#FFFF00")
        
        self.place_at_grid(v1_label, 'D3', scale_factor=0.6)
        self.place_at_grid(v2_label, 'B4', scale_factor=0.6)
        self.place_at_grid(b_label, 'C6', scale_factor=0.7)

        # Asset-based Parallelogram
        # Note: SVGAsset Mobject loading
        parallelogram = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/parallelogram.svg")
        parallelogram.set_fill(WHITE, opacity=0.2)
        parallelogram.set_stroke(WHITE, width=2)
        self.place_in_area(parallelogram, 'A2', 'C5', scale_factor=0.65)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF00FF"))
        self.play(Create(v1_obj), FadeIn(v1_label), Create(v2_obj), FadeIn(v2_label))
        self.play(FadeIn(parallelogram))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.play(Indicate(v1_obj), Indicate(v2_obj))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.play(Create(b_obj), FadeIn(b_label))
        self.play(Indicate(b_obj))
        self.wait(1)
