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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Chaos of Individuality", [
            "Individual data points are often wildly chaotic.",
            "Different distributions show unique, messy patterns.",
            "We calculate sample means to find stability."
        ])
        
        # Asset Loading (placeholder as none.svg is empty/generic)
        # Using SVGMobject as requested by logic, though '/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg' is just a placeholder
        dots_container = VGroup()
        for _ in range(500):
            dot = Dot(radius=0.03, color="#FF5733")
            dots_container.add(dot)

        # Place the data points as per instruction 21
        self.place_in_area(dots_container, 'A4', 'C6', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF5733")
        self.play(FadeIn(dots_container), run_time=1)
        
        label = Text("Data Points", font_size=24, color="#FFFFFF")
        self.place_at_grid(label, "A4") # Tethered to data points
        self.play(Write(label))
        
        # Animate movement
        def update_dots(mob, dt):
            for dot in mob:
                dot.shift(np.array([np.random.uniform(-0.02, 0.02), np.random.uniform(-0.02, 0.02), 0]))
                # Keep within bounds relative to the area
                dot.set_x(np.clip(dot.get_x(), 3.5, 6.5))
                dot.set_y(np.clip(dot.get_y(), 0.5, 2.5))

        dots_container.add_updater(update_dots)
        self.wait(3)
        dots_container.remove_updater(update_dots)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFC300")
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#DAF7A6")
        self.play(self.lecture[1].animate.set_color(WHITE))
        self.wait(2)
