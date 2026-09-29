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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["The sum is only valid for Re(s) > 1.", "Analytic continuation extends the function further.", "It reveals structure across the complex plane."]
        self.setup_layout("Analytical Continuation: The Magic of Symmetry", lecture_lines)
        
        # Load Assets
        ruler_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        magnifying_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifying.svg")
        
        # Prepare elements
        plane = ComplexPlane(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_numbers": False})
        self.place_in_area(plane, 'A2', 'D5', scale_factor=0.4)
        
        line_grp = self.lecture
        
        # === Animation for Lecture Line 1 ===
        # Draw complex plane with '#FFFFFF' using ruler asset
        self.place_at_grid(ruler_icon, 'A2', scale_factor=0.3)
        self.play(line_grp[0].animate.set_color("#FFFFFF"), Create(plane), FadeIn(ruler_icon), run_time=2)

        # === Animation for Lecture Line 2 ===
        curve = ParametricFunction(lambda t: np.array([t, 0.5 * np.sin(t*3), 0]), t_range=[-2, 2])
        curve.set_color("#FF69B4")
        self.place_at_grid(curve, 'C4', scale_factor=0.5)
        
        label = Text("Extension", font_size=16)
        self.place_at_grid(label, 'E4', scale_factor=0.4)
        
        self.play(line_grp[1].animate.set_color("#FF69B4"), Create(curve), Write(label), run_time=2)

        # === Animation for Lecture Line 3 ===
        # Show pole locations in '#FFFF00' using magnifying asset
        dots = VGroup(*[Dot(point=plane.c2p(x, y), color="#FFFF00") for x, y in [(-1, 0), (0, 0), (1, 1)]])
        self.place_at_grid(magnifying_icon, 'F5', scale_factor=0.3)
        self.play(line_grp[2].animate.set_color("#FFFF00"), FadeIn(dots), FadeIn(magnifying_icon), run_time=2)
        self.wait(2)
