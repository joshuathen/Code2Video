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
        self.setup_layout("Application and Summary", [
            "Determinants solve systems and simplify integration.",
            "Matrices represent transformations, determinants the scaling.",
            "These tools are fundamental in applied mathematics."
        ])
        
        # Assets
        calc_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        comp_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        prot_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")

        # === Animation for Lecture Line 1 ===
        # Summarize Determinant as scaling factor
        sum_text = Text("Determinant = Scaling Factor", color="#FFD700")
        self.place_in_area(sum_text, 'B2', 'B4', scale_factor=0.5)
        self.place_at_grid(calc_icon, 'B5', scale_factor=0.5)
        self.play(Write(sum_text), FadeIn(calc_icon))
        self.play(self.lecture[0].animate.set_color("#FFD700"))

        # === Animation for Lecture Line 2 ===
        # Show 2D vs 3D comparison table
        table = Table(
            [["2D", "Area"], ["3D", "Volume"]],
            col_labels=[Text("Dim"), Text("Scales")],
            include_outer_lines=True
        ).scale(0.4)
        self.place_at_grid(table, 'D5', scale_factor=0.7)
        self.place_at_grid(comp_icon, 'D2', scale_factor=0.5)
        self.play(Create(table), FadeIn(comp_icon))
        self.play(self.lecture[1].animate.set_color("#00CED1"))

        # === Animation for Lecture Line 3 ===
        # Fade out elements with a final text: 'Determinant'
        final_text = Text("Determinant", font_size=48, color=WHITE)
        self.place_at_grid(final_text, 'C5', scale_factor=0.9)
        self.place_at_grid(prot_icon, 'E2', scale_factor=0.5)
        
        self.play(
            FadeOut(sum_text),
            FadeOut(table),
            FadeOut(calc_icon),
            FadeOut(comp_icon),
            Write(final_text),
            FadeIn(prot_icon)
        )
        self.play(self.lecture[2].animate.set_color("#FF4500"))
        self.wait(2)
