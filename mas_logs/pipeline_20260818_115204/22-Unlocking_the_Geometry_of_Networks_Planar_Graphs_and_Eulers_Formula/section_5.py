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
        self.setup_layout("Summary & Application", ["Graph properties relate to duals.", "Duality aids network reliability analysis.", "It is essential for graph coloring."])
        
        # === Animation for Lecture Line 1 ===
        # Display summary list #FFFFFF
        list_obj = VGroup(
            Text("V, E, F -> V', E', F'", font_size=24, color=WHITE),
            Text("Duality Preservation", font_size=24, color=WHITE)
        ).arrange(DOWN, aligned_edge=LEFT)
        self.place_at_grid(list_obj, "B5", scale_factor=0.8)
        self.play(FadeIn(list_obj))
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show real-world application icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/satellite.svg] #00FFFF
        icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/satellite.svg", color="#00FFFF")
        label = Text("Network Reliability", font_size=24, color="#00FFFF")
        app_group = VGroup(icon, label).arrange(DOWN)
        self.place_at_grid(app_group, "E2", scale_factor=0.7)
        self.play(FadeIn(icon), Write(label))
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Final screen showing V-E+F=2 [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg] #FFFF00
        icon_comp = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg", color="#FFFF00")
        formula = MathTex("V - E + F = 2", color="#FFFF00")
        formula_group = VGroup(icon_comp, formula).arrange(DOWN)
        self.place_at_grid(formula_group, "F4", scale_factor=1.0)
        self.play(FadeIn(icon_comp), Write(formula))
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.wait(2)
