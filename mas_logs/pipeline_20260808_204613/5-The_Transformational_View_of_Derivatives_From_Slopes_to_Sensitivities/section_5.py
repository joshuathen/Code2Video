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
        self.setup_layout("Summary & Synthesis", ["Derivatives reveal linear truth in curves.", "We turn dynamic change into static ratios.", "Zooming in makes complexity manageable."])
        
        # === Animation for Lecture Line 1 ===
        # Display summary table of derivative concepts
        table = Table(
            [["Concept", "Role"], ["Velocity", "Tangent"], ["Sensitivity", "Linearity"]],
            include_outer_lines=True
        )
        self.place_in_area(table, 'A1', 'C6', scale_factor=0.6)
        
        graph_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/graph.svg")
        self.place_at_grid(graph_icon, 'A6', scale_factor=0.3)
        
        self.play(FadeIn(table), FadeIn(graph_icon))
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        # Animate connecting lines between 'Velocity', 'Tangent', and 'Linearity'
        v = table.get_cell((2, 1))
        t = table.get_cell((2, 2))
        l = table.get_cell((3, 2))
        
        line1 = Line(v.get_right(), t.get_left(), color="#FF5733")
        line2 = Line(t.get_bottom(), l.get_top(), color="#FF5733")
        
        self.play(Create(line1), Create(line2))
        self.lecture[1].set_color("#FF5733")
        
        # === Animation for Lecture Line 3 ===
        # Flash the synthesized conclusion 'Derivative = Local Sensitivity'
        conclusion = MathTex(r"\\text{Derivative} = \\text{Local Sensitivity}", color="#33FF57")
        self.place_in_area(conclusion, 'E1', 'F6', scale_factor=0.8)
        
        mag_glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifyingglass.svg")
        self.place_at_grid(mag_glass, 'E1', scale_factor=0.4)
        
        self.play(Write(conclusion), FadeIn(mag_glass))
        self.play(Indicate(conclusion))
        self.lecture[2].set_color("#33FF57")
        
        self.wait(2)
