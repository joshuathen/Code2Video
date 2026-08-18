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
        lecture_lines = ["Rates of change reveal original states.", "Integrals recover the position from acceleration.", "This solves complex real-world physical problems."]
        self.setup_layout("Application: From Rate to Reality", lecture_lines)
        
        # Setup plot axes
        axes = Axes(x_range=[0, 6, 1], y_range=[0, 4, 1], axis_config={"include_tip": False})
        self.place_in_area(axes, "A1", "D4", scale_factor=0.5)
        
        v_func = axes.plot(lambda t: 0.5 * t + 0.5, color=WHITE)
        
        # Load Assets
        vehicle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vehicle.svg")
        bridge = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg")
        
        # Displacement tracker
        disp_val = ValueTracker(0)
        disp_label = Text("Displacement: 0", font_size=24, color=YELLOW)
        self.place_at_grid(disp_label, "D5", scale_factor=0.8)
        
        summary_text = Text("Rocket landing precision", font_size=20, color=BLUE)
        self.place_in_area(summary_text, "E1", "F6", scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.place_at_grid(vehicle, "B6", scale_factor=0.3)
        self.play(Create(axes), Create(v_func), FadeIn(vehicle))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        area = axes.get_area(v_func, x_range=[0, 5], color="#00FF00", opacity=0.3)
        self.play(FadeIn(area))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.place_at_grid(bridge, "C5", scale_factor=0.3)
        self.play(FadeIn(bridge))
        
        # Update displacement label - Use internal state logic instead of always_redraw/lambda
        disp_label_text = Text("Displacement: 0.0", font_size=24, color=YELLOW)
        self.place_at_grid(disp_label_text, "D5", scale_factor=0.8)
        self.remove(disp_label)
        self.add(disp_label_text)
        
        def update_label(m):
            new_text = Text(f"Displacement: {disp_val.get_value():.1f}", font_size=24, color=YELLOW)
            self.place_at_grid(new_text, "D5", scale_factor=0.8)
            m.become(new_text)
            
        disp_label_text.add_updater(update_label)
        self.play(disp_val.animate.set_value(5.0), run_time=2)
        disp_label_text.remove_updater(update_label)
        self.wait(1)
