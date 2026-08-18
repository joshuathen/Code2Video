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
        self.setup_layout("Closure Problem & Mathematical Structure", [
            "Navier-Stokes non-linearity creates closure problem.",
            "We use statistical structure functions instead.",
            "Equations exhibit spatial and temporal self-similarity."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show hierarchy of momentum equations [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/pipe.svg] #00FFFF.
        pipe = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pipe.svg", color="#00FFFF")
        eq_group = VGroup(*[MathTex(f"U_{i} \\cdot \\nabla U_{i} = ...", font_size=24) for i in range(3)], pipe)
        eq_group.arrange(DOWN, buff=0.3)
        self.place_in_area(eq_group, 'A2', 'C5', scale_factor=0.9)
        
        self.play(FadeIn(eq_group))
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        # Highlight the closure break point [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/nozzle.svg] #FF0000.
        nozzle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/nozzle.svg", color="#FF0000")
        break_point = Circle(radius=0.3, color="#FF0000").set_stroke(width=4)
        group_2 = VGroup(nozzle, break_point).arrange(DOWN)
        self.place_at_grid(group_2, 'D5', scale_factor=1.0)
        
        self.play(Create(group_2))
        self.lecture[1].set_color("#FF0000")

        # === Animation for Lecture Line 3 ===
        # Simplify with modeling assumptions [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/fluid.svg] #FFFF00.
        fluid = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fluid.svg", color="#FFFF00")
        rect = RoundedRectangle(corner_radius=0.1, color="#FFFF00", height=1.0, width=2.0)
        label = Text("Self-Similarity Model", font_size=18, color="#FFFF00")
        model = VGroup(fluid, rect, label).arrange(DOWN)
        self.place_in_area(model, 'E3', 'F6', scale_factor=0.9)
        
        self.play(Write(model))
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)
