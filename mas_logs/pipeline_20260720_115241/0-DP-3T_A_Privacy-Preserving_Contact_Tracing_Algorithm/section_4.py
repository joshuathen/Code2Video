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

class Section4Scene(TeachingScene):
    def construct(self):
        title = "The \"Ephemeral\" Advantage: Mitigating Long-Term Tracking"
        lecture_lines = [
            "Ephemeral IDs change frequently.",
            "This prevents long-term tracking.",
            "No permanent record of contacts is made.",
            "It stops profiling and correlation of events.",
            "Privacy is maintained through short ID lifespans."
        ]
        self.setup_layout(title, lecture_lines)

        # Define colors for lecture lines
        colors = [
            "#00FFFF",  # Cyan for line 1
            "#FF00FF",  # Magenta for line 2
            "#FFFF00",  # Yellow for line 3
            "#FFA500",  # Orange for line 4
            "#FF4500"   # OrangeRed for line 5
        ]

        # === Animation for Lecture Line 1 ===
        # Animate data stored on a device with a timer.
        device_data = Text("Device Data", font_size=24).move_to(self.grid["C3"])
        timer = Text("Timer", font_size=24).next_to(device_data, DOWN)
        data_timer_group = VGroup(device_data, timer)
        
        self.play(Write(data_timer_group))
        self.play(Indicate(timer))
        self.lecture[0].set_color(colors[0])
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show the timer running out and the data disappearing.
        timer_running_out = timer.copy().set_color(RED)
        self.play(Transform(timer, timer_running_out))
        self.play(FadeOut(device_data), FadeOut(timer))
        self.lecture[1].set_color(colors[1])
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Contrast this with a visual of a long-term data archive.
        long_term_archive_label = Text("Long-Term Archive", font_size=24).move_to(self.grid["C3"])
        long_term_data = Rectangle(width=2, height=1, fill_opacity=0.5, color=GRAY).next_to(long_term_archive_label, DOWN)
        archive_group = VGroup(long_term_archive_label, long_term_data)
        self.play(Write(archive_group))
        self.lecture[2].set_color(colors[2])
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Emphasize that no single device holds long-term infection history.
        # We can represent this by showing the archive is vast but disconnected from any single device.
        no_history_text = Text("No single device holds history", font_size=20, color=YELLOW).move_to(self.grid["E4"])
        self.play(Write(no_history_text))
        self.lecture[3].set_color(colors[3])
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Conclude that this ephemeral nature prevents long-term tracking.
        privacy_maintained = Text("Privacy Maintained", font_size=24, color=GREEN).move_to(self.grid["C5"])
        self.play(Write(privacy_maintained))
        self.lecture[4].set_color(colors[4])
        self.wait(2)

        # Fade out all elements except the title and lecture lines
        self.play(
            FadeOut(data_timer_group), 
            FadeOut(archive_group),
            FadeOut(no_history_text),
            FadeOut(privacy_maintained)
        )
        self.wait(1)
