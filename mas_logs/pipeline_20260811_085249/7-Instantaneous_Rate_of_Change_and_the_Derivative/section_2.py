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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Transition to Instantaneous Rate", ["Zoom in on the curve.", "Secant becomes a tangent line.", "As Δt shrinks, rate stabilizes."])
        
        # Elements
        curve = FunctionGraph(lambda x: 0.1 * x**3, x_range=[-3, 3], color=BLUE)
        self.place_in_area(curve, 'A3', 'F6', scale_factor=0.8)
        self.add(curve)
        
        point_a = Dot(color="#00FF00")
        self.place_at_grid(point_a, 'D3', scale_factor=0.7)
        self.add(point_a)
        
        point_b = Dot(color=YELLOW)
        self.place_at_grid(point_b, 'C5', scale_factor=0.7)
        self.add(point_b)
        
        # Asset: Magnifying glass
        magnifier = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifying-glass.svg")
        
        secant = Line(point_a.get_center(), point_b.get_center(), color=RED)
        self.add(secant)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FF00")
        self.place_at_grid(magnifier, "C3", scale_factor=0.5)
        self.play(FadeIn(magnifier))
        self.play(Indicate(point_a))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        # Move point_b towards A to represent secant -> tangent
        self.play(
            point_b.animate.move_to(point_a.get_center() + np.array([0.2, 0.05, 0])),
            UpdateFromAlphaFunc(secant, lambda m, a: m.put_start_and_end_on(point_a.get_center(), point_b.get_center())),
            magnifier.animate.move_to(point_a.get_center()),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(ORANGE)
        self.play(FadeOut(magnifier), Indicate(secant))
        self.wait(2)
