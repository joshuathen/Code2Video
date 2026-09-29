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
        lecture_lines = [
            "Winding number counts loops around points.",
            "Ant walking around trees defines paths.",
            "Circling roots yields non-zero counts."
        ]
        self.setup_layout("Winding Numbers: Counting the Loops", lecture_lines)
        
        # Elements
        plane = ComplexPlane(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_numbers": False})
        self.place_in_area(plane, 'B3', 'E6', scale_factor=0.9)
        
        pole = Dot(color=RED)
        pole.move_to(plane.c2p(0, 0))
        
        curve = ParametricFunction(
            lambda t: plane.c2p(np.cos(t) + 0.5*np.cos(2*t), np.sin(t) + 0.5*np.sin(2*t)),
            t_range=[0, 2*PI], color=BLUE
        )
        
        # Keep Winding Number text fixed to avoid flickering/jumping, but it must update color/value
        wn_text = Text("N = 0", font_size=32, color=YELLOW)
        self.place_at_grid(wn_text, 'E4', scale_factor=1.2)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(Create(plane), Create(curve), Create(pole))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(GREEN))
        ant = Dot(color=YELLOW).move_to(curve.point_from_proportion(0))
        self.play(MoveAlongPath(ant, curve), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(RED))
        
        # Replace the text object safely as per instructions
        new_wn_text = Text("N = 1", font_size=32, color=RED)
        new_wn_text.move_to(wn_text.get_center())
        self.play(Transform(wn_text, new_wn_text))
        self.wait(2)
