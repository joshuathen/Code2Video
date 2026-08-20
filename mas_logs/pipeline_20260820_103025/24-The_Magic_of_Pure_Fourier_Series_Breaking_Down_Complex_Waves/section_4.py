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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Visualizing Convergence: The Square Wave", [
            "Adding harmonics builds complex shapes.",
            "Start with one simple sine wave.",
            "More terms create sharper edges.",
            "Watch the robotic hand trace corners.",
            "This is how functions converge."
        ])

        # Prepare wave functions
        axes = Axes(x_range=[-PI, PI], y_range=[-1.5, 1.5], axis_config={"include_tip": False})
        # Applying requested position fixes
        self.place_in_area(axes, 'C2', 'D6', scale_factor=0.6)

        # Load robotic hand asset
        hand = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hand.svg")
        self.place_at_grid(hand, "F4", scale_factor=0.4)
        hand.set_opacity(0) # Initially invisible

        def get_wave(n_terms):
            return axes.plot(lambda x: sum([ (4/(np.pi*(2*k-1))) * np.sin((2*k-1)*x) for k in range(1, n_terms+1) ]), color=WHITE)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFFFF")
        wave1 = axes.plot(lambda x: np.sin(x), color=WHITE)
        self.play(Create(wave1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFD700")
        wave3 = axes.plot(lambda x: np.sin(x) + (1/3)*np.sin(3*x), color="#FFD700")
        self.play(Transform(wave1, wave3))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#D8BFD8")
        wave7 = axes.plot(lambda x: sum([(4/(np.pi*(2*k-1))) * np.sin((2*k-1)*x) for k in range(1, 5)]), color="#D8BFD8")
        self.play(FadeIn(hand), Transform(wave1, wave7))
        self.play(hand.animate.move_to(axes.coords_to_point(PI/2, 1.2))) # Point to corner
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#40E0D0")
        final_wave = axes.plot(lambda x: sum([(4/(np.pi*(2*k-1))) * np.sin((2*k-1)*x) for k in range(1, 20)]), color="#40E0D0")
        self.play(Transform(wave1, final_wave), FadeOut(hand), run_time=2)
        self.wait(2)
