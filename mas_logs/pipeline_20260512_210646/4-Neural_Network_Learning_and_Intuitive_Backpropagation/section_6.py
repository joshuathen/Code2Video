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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup layout with title and expanded lecture lines
        self.setup_layout(
            "The Result: Iteration and Convergence", 
            [
                "Training involves repeating this process many thousands of times.", 
                "The total error drops closer to zero each epoch.", 
                "Nero finally masters the patterns within the data.", 
                "He identifies the dog with high confidence.", 
                "The model is now fully trained and accurate."
            ]
        )
        
        dog_asset = "/mmfs1/data/home/jthen/Code2Video/assets/icon/dog.svg"
        nero_asset = "/mmfs1/data/home/jthen/Code2Video/assets/icon/nero.svg"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        # Create a representation of Nero's Neural Network
        net_box = RoundedRectangle(corner_radius=0.2, width=1.8, height=1.5, color=BLUE_B)
        net_label = Text("Nero's Net", font_size=18, color=BLUE_B)
        network = VGroup(net_box, net_label)
        self.place_at_grid(network, "B3")
        self.play(FadeIn(network))
        
        # Fast sequence of images [Asset: dog.svg] passing through
        for _ in range(4):
            img_icon = SVGMobject(dog_asset, color=WHITE)
            # Use B2 and scale 0.8 for better margin (Issue 58 & 65)
            self.place_at_grid(img_icon, "B2", scale_factor=0.8)
            
            self.play(
                img_icon.animate.move_to(self.grid["B3"]),
                run_time=0.15,
                rate_func=linear
            )
            self.play(
                img_icon.animate.move_to(self.grid["B4"]).set_opacity(0),
                run_time=0.15,
                rate_func=linear
            )
            self.remove(img_icon)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        
        # Plot a green line graph decreasing toward zero
        axis_x = Line(self.grid["F4"], self.grid["F6"], color=WHITE)
        axis_y = Line(self.grid["F4"], self.grid["D4"], color=WHITE)
        
        # Position graph_label at E5 with scale 0.8 (Issue 57 & 65)
        graph_label = Text("Loss", font_size=16, color="#00FF00")
        self.place_at_grid(graph_label, "E5", scale_factor=0.8)
        
        # Convergence curve
        p1 = self.grid["D4"] + UP*0.4
        p2 = self.grid["E5"] + UP*0.1
        p3 = self.grid["F6"] + UP*0.2
        convergence_curve = VMobject(color="#00FF00")
        convergence_curve.set_points_as_corners([p1, p2, p3])
        convergence_curve.make_smooth()
        
        self.play(Create(axis_x), Create(axis_y), FadeIn(graph_label))
        self.play(Create(convergence_curve), run_time=2.0)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        
        # Show Nero [Asset: nero.svg] at B5 (happy expression color #00FF00)
        nero = SVGMobject(nero_asset, color="#00FF00")
        self.place_at_grid(nero, "B5", scale_factor=1.0)
        self.play(FadeIn(nero))
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FF00")
        
        # Display '99% Dog [Asset: dog.svg]' scale 0.7 at C5 (Issue 59 & 65)
        conf_val = Text("99% Dog", font_size=20, color="#00FF00", weight=BOLD)
        conf_icon = SVGMobject(dog_asset, color="#00FF00")
        confidence_group = VGroup(conf_val, conf_icon).arrange(RIGHT, buff=0.15)
        self.place_at_grid(confidence_group, "C5", scale_factor=0.7)
        
        self.play(Write(confidence_group))
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFFFF")
        
        # Fade in 'Model Trained' banner (#FFFFFF)
        banner_rect = Rectangle(width=3.5, height=0.6, fill_color=BLACK, fill_opacity=0.8, stroke_color=WHITE)
        banner_text = Text("MODEL TRAINED", font_size=22, color=WHITE, weight=BOLD)
        banner_grp = VGroup(banner_rect, banner_text)
        self.place_in_area(banner_grp, "A3", "A4")
        
        self.play(FadeIn(banner_grp))
        self.wait(3)
