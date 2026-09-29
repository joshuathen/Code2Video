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
        lecture_lines = [
            "Music starts with a steady, recurring pulse.",
            "Think of a heartbeat, always consistent.",
            "These pulses provide our musical foundation."
        ]
        self.setup_layout("Prerequisite: The Concept of a Steady Pulse", lecture_lines)
        
        # Heart asset path
        heart_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/heart.svg"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FF00")
        
        # Create initial dot as placeholder before heartbeat asset is introduced
        pulse_dot = Dot(color="#00FF00", radius=0.3)
        self.place_at_grid(pulse_dot, "D2", scale_factor=0.8)
        self.add(pulse_dot)
        self.wait(4)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FF00")
        
        # Replace dot with heart asset
        heart_asset = SVGMobject(heart_path)
        heart_asset.set_color(RED)
        self.place_at_grid(heart_asset, "D2", scale_factor=0.6)
        
        self.remove(pulse_dot)
        self.add(heart_asset)
        
        # Heartbeat pulse animation using ValueTracker
        tracker = ValueTracker(1.0)
        heart_asset.add_updater(lambda m: m.set_height(1.0 * tracker.get_value()))
        
        for _ in range(4):
            tracker.set_value(1.2)
            self.wait(0.5)
            tracker.set_value(1.0)
            self.wait(0.5)
            
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FF00")
        
        # Add label
        pulse_label = Text("Pulse", font_size=32, color=WHITE)
        self.place_at_grid(pulse_label, "D3", scale_factor=0.8)
        self.play(FadeIn(pulse_label))
        self.wait(4)
