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
        lines = [
            "Algorithms don't search for the secret word.",
            "They prioritize the most informative guess instead.",
            "The bot tames uncertainty using entropy metrics.",
            "Every outcome distribution is analyzed for gain.",
            "Entropy maximization is the key strategy here."
        ]
        self.setup_layout("The Strategy: Maximizing Entropy", lines)
        
        # === Animation for Lecture Line 1 ===
        # Show an 'Entropy Maximizer' slider, use SVG asset
        slider = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/slider.svg", color="#00FF00")
        label_ent = Text("Entropy", font_size=20, color="#00FF00").next_to(slider, UP)
        slider_container = VGroup(slider, label_ent)
        self.place_at_grid(slider_container, 'A4', scale_factor=1.0)
        self.add(slider_container)
        self.lecture[0].set_color("#00FF00")
        self.wait(4)

        # === Animation for Lecture Line 2 ===
        # Slider moves to 'Maximum Uncertainty'
        self.play(slider.animate.move_to(self.grid['B5']), run_time=2)
        self.lecture[1].set_color("#00FF00")
        self.wait(4)

        # === Animation for Lecture Line 3 ===
        # Display 'Entropy H(X)' graph
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 2, 1], x_length=3, y_length=2, axis_config={"include_tip": False})
        curve = axes.plot(lambda x: -(x/4)*np.log2((x/4)+1e-5), color="#FFFF00")
        label_hx = Text("H(X)", font_size=20, color="#FFFF00")
        
        graph = VGroup(axes, curve)
        self.place_in_area(graph, 'B4', 'D6', scale_factor=0.9)
        self.place_at_grid(label_hx, 'D5', scale_factor=0.7)
        self.add(graph, label_hx)
        self.lecture[2].set_color("#FFFF00")
        self.wait(4)

        # === Animation for Lecture Line 4 ===
        # Terminal Target icon on entropy curve
        target = Dot(color="#00FF00", radius=0.1)
        target_outline = Circle(radius=0.2, color="#00FF00")
        terminal_icon = VGroup(target, target_outline)
        terminal_icon.move_to(axes.c2p(2, 1.5))
        self.add(terminal_icon)
        self.lecture[3].set_color("#00FF00")
        self.wait(4)

        # === Animation for Lecture Line 5 ===
        # Flash label, use slider reference
        self.lecture[4].set_color("#FFFF00")
        self.play(Indicate(slider), Indicate(terminal_icon), run_time=2)
        self.wait(4)
