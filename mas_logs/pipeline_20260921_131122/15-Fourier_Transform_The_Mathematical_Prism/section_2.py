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
        self.setup_layout("Prerequisite: The Language of Waves", [
            "Periodic functions repeat cycles over time.",
            "Time domain shows events as they occur.",
            "Frequency domain reveals a signal's composition."
        ])
        
        # Load Assets
        ocean_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ocean.svg")
        tuning_fork_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tuningfork.svg")

        # === Animation for Lecture Line 1 ===
        # Periodic functions repeat cycles over time.
        self.lecture[0].set_color("#00FFFF")
        
        wave = FunctionGraph(lambda t: 0.5 * np.sin(2 * PI * t), x_range=[-2, 2], color="#00FFFF")
        self.place_in_area(wave, "B2", "D5", scale_factor=0.6)
        self.place_at_grid(ocean_icon, "A1", scale_factor=0.5)
        
        self.play(Create(wave), FadeIn(ocean_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Time domain shows events as they occur.
        self.lecture[1].set_color("#FF00FF")
        
        peak_label = Text("Crest", font_size=20, color="#FF00FF")
        valley_label = Text("Trough", font_size=20, color="#FFFF00")
        
        self.place_at_grid(peak_label, "B4", scale_factor=0.7)
        self.place_at_grid(valley_label, "D4", scale_factor=0.7)
        
        self.play(Write(peak_label), Write(valley_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Frequency domain reveals a signal's composition.
        self.lecture[2].set_color("#FFFFFF")
        
        wave2 = FunctionGraph(lambda t: 0.3 * np.sin(4 * PI * t), x_range=[-2, 2], color="#FFFFFF")
        self.place_at_grid(tuning_fork_icon, "F6", scale_factor=0.5)
        
        self.play(Transform(wave, wave2), FadeIn(tuning_fork_icon), run_time=1.5)
        self.wait(2)
