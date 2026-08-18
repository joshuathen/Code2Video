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
        lecture_lines = [
            "The network is an associative memory system.",
            "Inputs map directly to factual completions.",
            "MLP neurons act as the knowledge retrieval.",
            "Patterns trigger the storage of specific facts.",
            "This structure enables efficient fact retrieval."
        ]
        self.setup_layout("Synthesis and Summary", lecture_lines)
        
        # Create Visuals
        # Recap Icon Set - incorporating neuron asset
        neuron_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/neuron.svg", color="#FFFF00")
        recap_icons = VGroup(
            neuron_icon,
            Square(color="#FFFF00", fill_opacity=0.5, side_length=0.4),
            Circle(color="#FFFF00", fill_opacity=0.5, radius=0.2)
        ).arrange(RIGHT, buff=0.2)
        self.place_at_grid(recap_icons, 'B4', scale_factor=0.6)
        recap_icons.set_opacity(0)
        
        # Final Summary text
        final_text = Text("Knowledge Retrieved!", font_size=36, color=WHITE)
        self.place_in_area(final_text, 'E2', 'F5', scale_factor=0.8)
        final_text.set_opacity(0)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFCCCC"), run_time=0.5)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#CCFFCC"), run_time=0.5)
        self.play(FadeIn(recap_icons), run_time=1.0)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#CCCCFF"), run_time=0.5)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFFCC"), run_time=0.5)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#CCFFFF"), run_time=0.5)
        self.play(FadeIn(final_text), run_time=1.0)
        
        self.wait(2)
        self.play(FadeOut(self.lecture), FadeOut(recap_icons), FadeOut(final_text), run_time=1.0)
        self.wait(1)
