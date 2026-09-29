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
        self.setup_layout("Prerequisite & The 2D 'Pseudo-Scalar'", [
            "2D cross product measures signed parallelogram area.",
            "It indicates rotation direction: clockwise or counter-clockwise.",
            "It represents the perpendicularity of two vectors."
        ])
        
        # Setup Coordinate Plane and Vectors
        axes = Axes(x_range=[-2, 4], y_range=[-2, 4], axis_config={"include_tip": True}).scale(0.6)
        # Fix 17/29
        self.place_in_area(axes, "B2", "E5", scale_factor=0.9)
        
        u = Vector([2, 1], color="#FF6666")
        v = Vector([1, 2], color="#66CCFF")
        
        # Position them relative to origin
        u.shift(axes.c2p(0, 0) - u.get_start())
        v.shift(axes.c2p(0, 0) - v.get_start())
        
        u_label = MathTex(r"\vec{u}", color="#FF6666").next_to(u.get_end(), RIGHT)
        v_label = MathTex(r"\vec{v}", color="#66CCFF").next_to(v.get_end(), UP)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/parallelogram.svg]
        # Using SVG asset for parallelogram
        try:
            parallelogram = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/parallelogram.svg")
            parallelogram.set_color("#FFFF66")
            parallelogram.set_fill(opacity=0.3)
            # Position it roughly over the parallelogram formed by u and v
            # The vectors go from (0,0) to (2,1) and (1,2)
            # The parallelogram centroid is (0.5*(2+1), 0.5*(1+2)) = (1.5, 1.5) in data coords
            parallelogram.move_to(axes.c2p(1.5, 1.5))
            parallelogram.scale(0.5) 
        except Exception:
            # Fallback
            parallelogram = Polygon(
                axes.c2p(0, 0),
                axes.c2p(2, 1),
                axes.c2p(3, 3),
                axes.c2p(1, 2),
                color="#FFFF66",
                fill_opacity=0.3
            )
        
        formula = MathTex(r"\text{Area} = |u_1v_2 - u_2v_1|", color=WHITE)
        # Fix 18/30
        self.place_at_grid(formula, "E3", scale_factor=1.0)
        
        area_label = Text("Area", font_size=20, color=WHITE)
        # Fix 19/31
        self.place_at_grid(area_label, "D3", scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(axes), Create(u), Write(u_label), Create(v), Write(v_label))
        self.lecture[0].set_color("#FFFF66")
        
        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(parallelogram))
        self.lecture[1].set_color("#FFFF66")
        
        # === Animation for Lecture Line 3 ===
        self.play(Write(formula), FadeIn(area_label))
        self.lecture[2].set_color("#FFFF66")
        
        self.wait(2)
