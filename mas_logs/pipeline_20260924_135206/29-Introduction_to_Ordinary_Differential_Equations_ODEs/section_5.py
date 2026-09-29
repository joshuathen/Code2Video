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
        self.setup_layout("Summary & Real-World Application", [
            "ODEs are the language of dynamic systems.",
            "From pendulums to orbits, we predict change.",
            "Mastering them unlocks the future state."
        ])
        
        # Icons as SVGMobjects
        pendulum = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pendulum.svg")
        circuit = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circuit.svg")
        orbit = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/planet.svg")
        
        icons = VGroup(pendulum, circuit, orbit)
        
        # Positioning based on VideoCritic feedback
        self.place_at_grid(pendulum, "C2", scale_factor=0.9)
        self.place_at_grid(circuit, "C3", scale_factor=0.9)
        self.place_at_grid(orbit, "C4", scale_factor=0.9)
        
        # Overlay Text
        prediction_text = Text("Predicting Dynamic Systems", font_size=24, color=WHITE)
        self.place_in_area(prediction_text, "E2", "E5", scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE), FadeIn(pendulum))
        # Simulated swinging for pendulum
        self.play(Rotate(pendulum, angle=PI/6, about_point=pendulum.get_top(), run_time=1.5))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN), FadeIn(circuit), FadeIn(orbit))
        # Simulated glow for circuit and rotation for orbit
        self.play(Indicate(circuit), Rotate(orbit, angle=2*PI, run_time=1.5))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW), Write(prediction_text))
        self.play(Flash(icons, color=WHITE), Flash(prediction_text, color=WHITE))
        
        # FIX: Using FullScreenRectangle as FullScreenFadeRectangle is not a core Manim class
        fade_rect = FullScreenRectangle(fill_opacity=1.0, color=BLACK)
        self.add(fade_rect)
        self.play(FadeIn(fade_rect, run_time=1.0))
        self.wait(1)
