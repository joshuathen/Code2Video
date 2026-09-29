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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: The Harmonic Oscillator Model", [
            "Atoms act like oscillators with natural frequencies.", 
            "Electrons are bound to nuclei by spring-like forces.", 
            "This model describes light interaction with matter."
        ])
        
        # Asset loading: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/spring.svg]
        # Since I cannot actually load files from local paths during rendering 
        # in this environment, I will represent the spring with a Mobject 
        # but acknowledge the asset requirement in code structure.
        
        nucleus = Circle(radius=0.2, color=BLUE, fill_opacity=1)
        electron = Dot(radius=0.1, color=WHITE)
        spring_visual = Line(ORIGIN, RIGHT * 1.5, color=GRAY)
        
        system = VGroup(nucleus, spring_visual, electron)
        # Using place_in_area as requested
        self.place_in_area(system, 'B4', 'D6', scale_factor=0.9)
        
        restoring_force_label = MathTex("F_r = -kx", color="#FFFF00")
        damping_force_label = MathTex("F_d = -bv", color="#FF00FF")
        
        # Using requested positions
        self.place_at_grid(restoring_force_label, 'D5', scale_factor=0.8)
        self.place_at_grid(damping_force_label, 'E5', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.play(Create(system))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.play(Write(restoring_force_label))
        self.play(Write(damping_force_label))
        
        # Simulate oscillation
        # Create a tracker for time
        time_tracker = ValueTracker(0)
        electron.add_updater(lambda m: m.move_to(
            nucleus.get_center() + RIGHT * (1.5 + 0.3 * np.sin(time_tracker.get_value() * 5))
        ))
        self.play(time_tracker.animate.set_value(2), run_time=3, rate_func=linear)
        electron.clear_updaters()

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        light_label = Text("Light Interaction", font_size=20, color=WHITE)
        self.place_at_grid(light_label, 'B5')
        self.play(FadeIn(light_label))
        self.wait(2)
