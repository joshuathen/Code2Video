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
        # Data from storyboard and outline
        title = "Vectors as Movements"
        lines = [
            "Think of a vector as a movement through space.",
            "An arrow starting at the origin shows the displacement.",
            "We can stretch or shrink this arrow through scaling."
        ]
        
        self.setup_layout(title, lines)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # Create coordinate grid
        plane = NumberPlane(
            x_range=[-1, 7, 1],
            y_range=[-1, 5, 1],
            background_line_style={
                "stroke_color": "#FFFFFF",
                "stroke_width": 1,
                "stroke_opacity": 0.3
            },
            axis_config={"include_tip": True, "color": "#FFFFFF"}
        )
        # Fix: Issue 32 - Reposition plane
        self.place_in_area(plane, 'C2', 'F6', scale_factor=0.7)
        
        # Vector (3,2)
        vector = plane.get_vector([3, 2, 0], color="#00FF00")
        
        # Asset: Vehicle
        vehicle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vehicle.svg")
        vehicle.set_color("#00FF00")
        vehicle.scale(0.3)
        # Position vehicle at the tip of the vector
        vehicle.move_to(vector.get_end())
        
        # Tracker for the tip position to keep vehicle attached
        tip_pos = ValueTracker(3.0)
        y_pos = ValueTracker(2.0)
        
        def vehicle_updater(obj):
            # Using the vector's current tip
            obj.move_to(vector.get_end())
            
        vehicle.add_updater(vehicle_updater)
        
        self.play(Create(plane), run_time=1.0)
        self.play(GrowArrow(vector), FadeIn(vehicle), run_time=1.5)
        
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FF0000")
        )
        
        # Components
        h_comp = Arrow(
            plane.c2p(0, 0), plane.c2p(3, 0), 
            buff=0, color="#FF0000", stroke_width=4
        )
        label_3 = MathTex("3", color="#FF0000", font_size=24)
        # Fix: Issue 32 - Place label_3 at F4
        self.place_at_grid(label_3, 'F4', scale_factor=0.6)
        
        v_comp = Arrow(
            plane.c2p(3, 0), plane.c2p(3, 2), 
            buff=0, color="#0000FF", stroke_width=4
        )
        label_2 = MathTex("2", color="#0000FF", font_size=24)
        # Fix: Issue 32 - Place label_2 at D6
        self.place_at_grid(label_2, 'D6', scale_factor=0.6)
        
        self.play(Create(h_comp), Write(label_3), run_time=1.0)
        self.play(Create(v_comp), Write(label_2), run_time=1.0)
        
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#00FF00"),
            FadeOut(h_comp), FadeOut(label_3),
            FadeOut(v_comp), FadeOut(label_2)
        )
        
        # Scaling animations
        # Scale to (6, 4)
        target_vec_large = plane.get_vector([6, 4, 0], color="#00FF00")
        
        self.play(
            Transform(vector, target_vec_large),
            run_time=2.0
        )
        self.wait(1.0)
        
        # Scale back to (3, 2)
        target_vec_small = plane.get_vector([3, 2, 0], color="#00FF00")
        
        self.play(
            Transform(vector, target_vec_small),
            run_time=2.0
        )
        
        vehicle.remove_updater(vehicle_updater)
        self.wait(2.0)
